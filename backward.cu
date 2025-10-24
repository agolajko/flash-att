#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void backward_kernel(
    const float *Q, const float *K, const float *V,
    const float *O, const float *dO,
    const float *l, const float *m,
    const int N, const int d,
    const int Tc, const int Tr,
    const int Bc, const int Br,
    const float softmax_scale,
    float *dQ, float *dK, float *dV)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; // batch and head index

    // Offset into Q,K,V,O - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);          // offset for l and m

    // Define SRAM for Q,K,V,S,dO,dQ,dK,dV
    extern __shared__ float sram[];
    int tile_size = Bc * d; // size of Qi, Kj, Vj

    float *Qi = sram;
    float *Kj = &sram[tile_size];
    float *Vj = &sram[tile_size * 2];
    float *dOi = &sram[tile_size * 3];
    float *dQi = &sram[tile_size * 4];

    // S and related matrices
    float *S = &sram[tile_size * 5];
    float *P = &sram[tile_size * 5 + Bc * Br]; // reuse same space as needed
    float *dS = &sram[tile_size * 5 + Bc * Br * 2];

    // Outer loop over blocks of K and V
    for (int j = 0; j < Tc; j++)
    {

        // Load Kj, Vj to SRAM (Algorithm 4, line 7)
        for (int x = 0; x < d; x++)
        {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }

        // Initialize dKj, dVj to 0 in SRAM (Algorithm 4, line 8)
        float dKj[64] = {0}; // assuming d <= 64, adjust as needed
        float dVj[64] = {0};

        __syncthreads();

        // Inner loop over blocks of Q
        for (int i = 0; i < Tr; i++)
        {

            // Load Qi, Oi, dOi, li, mi from HBM (Algorithm 4, line 10)
            for (int x = 0; x < d; x++)
            {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
                dOi[(tx * d) + x] = dO[qkv_offset + (tile_size * i) + (tx * d) + x];
            }

            float row_m = m[lm_offset + (Br * i) + tx];
            float row_l = l[lm_offset + (Br * i) + tx];

            // Load dQi (will be accumulated)
            for (int x = 0; x < d; x++)
            {
                dQi[(tx * d) + x] = dQ[qkv_offset + (tile_size * i) + (tx * d) + x];
            }

            __syncthreads();

            // Compute S_ij = softmax_scale * Q_i @ K_j^T (Algorithm 4, line 11)
            for (int y = 0; y < Bc; y++)
            {
                float sum = 0;
                for (int x = 0; x < d; x++)
                {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                S[(Bc * tx) + y] = sum * softmax_scale;
            }

            // Compute P_ij = diag(l_i)^-1 * exp(S_ij - m_i) (Algorithm 4, line 13)
            for (int y = 0; y < Bc; y++)
            {
                P[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m) / row_l;
            }

            // Compute D_i = rowsum(dO_i ⊙ O_i) (Algorithm 4, line 19)
            float Di = 0;
            for (int x = 0; x < d; x++)
            {
                float Oi_val = O[qkv_offset + (tile_size * i) + (tx * d) + x];
                Di += dOi[(tx * d) + x] * Oi_val;
            }

            // Compute dS_ij = P_ij ⊙ (dP_ij - D_i) (Algorithm 4, line 20)
            // First compute dP_ij = dO_i @ V_j^T (Algorithm 4, line 17)
            for (int y = 0; y < Bc; y++)
            {
                float dP_val = 0;
                for (int x = 0; x < d; x++)
                {
                    dP_val += dOi[(tx * d) + x] * Vj[(y * d) + x];
                }
                dS[(Bc * tx) + y] = P[(Bc * tx) + y] * (dP_val - Di);
            }

            __syncthreads();

            // Update dQ_i += softmax_scale * dS_ij @ K_j (Algorithm 4, line 21)
            for (int x = 0; x < d; x++)
            {
                float dq_update = 0;
                for (int y = 0; y < Bc; y++)
                {
                    dq_update += dS[(Bc * tx) + y] * Kj[(y * d) + x];
                }
                dQi[(tx * d) + x] += softmax_scale * dq_update;
            }

            // Write dQ_i back to HBM (Algorithm 4, line 21)
            for (int x = 0; x < d; x++)
            {
                dQ[qkv_offset + (tile_size * i) + (tx * d) + x] = dQi[(tx * d) + x];
            }

            // Accumulate dK_j += softmax_scale * dS_ij^T @ Q_i (Algorithm 4, line 22)
            // Note: each thread accumulates for its corresponding row of Kj
            for (int x = 0; x < d; x++)
            {
                float dk_update = 0;
                for (int y = 0; y < Br; y++)
                {
                    dk_update += dS[(Bc * y) + tx] * Qi[(y * d) + x];
                }
                dKj[x] += softmax_scale * dk_update;
            }

            // Accumulate dV_j += P_ij^T @ dO_i (Algorithm 4, line 16)
            for (int x = 0; x < d; x++)
            {
                float dv_update = 0;
                for (int y = 0; y < Br; y++)
                {
                    dv_update += P[(Bc * y) + tx] * dOi[(y * d) + x];
                }
                dVj[x] += dv_update;
            }

            __syncthreads();
        }

        // Write dK_j, dV_j to HBM (Algorithm 4, line 24)
        for (int x = 0; x < d; x++)
        {
            dK[qkv_offset + (tile_size * j) + (tx * d) + x] = dKj[x];
            dV[qkv_offset + (tile_size * j) + (tx * d) + x] = dVj[x];
        }

        __syncthreads();
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor l,
    torch::Tensor m)
{
    const int Bc = 32;
    const int Br = 32;

    const int B = Q.size(0);
    const int nh = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Tc = ceil((float)N / Bc);
    const int Tr = ceil((float)N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize dQ, dK, dV to zeros in HBM
    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);

    // Calculate SRAM size needed per block
    // Need space for: Qi, Kj, Vj, dOi, dQi, S, P, dS
    const int sram_size = (5 * Bc * d * sizeof(float)) + (3 * Bc * Br * sizeof(float));

    dim3 grid_dim(B, nh); // batch_size x num_heads
    dim3 block_dim(Bc);   // Bc threads per block

    backward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), dO.data_ptr<float>(),
        l.data_ptr<float>(), m.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>());

    return std::make_tuple(dQ, dK, dV);
}