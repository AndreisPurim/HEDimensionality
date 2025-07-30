#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include "../include/Client.h"
#include "../include/Math.h"

using namespace std;
typedef chrono::high_resolution_clock Clock;

vector<uint8_t> load_sample(const string& path, int target_sample_id, uint32_t nslots) {
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "[HEDim main.cpp] ERROR: Error opening file " << path << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        int sample_id, user_id;
        iss >> sample_id >> user_id;

        if (sample_id == target_sample_id) {
            vector<uint8_t> vec(nslots);
            for (uint32_t i = 0; i < nslots; ++i) {
                int val;
                if (!(iss >> val)) {
                    cerr << "[HEDim main.cpp] ERROR: Not enough values in vector for sample " << sample_id << endl;
                    exit(EXIT_FAILURE);
                }
                vec[i] = static_cast<uint8_t>(val);
            }
            return vec;
        }
    }

    cerr << "[HEDim main.cpp] ERROR: Sample ID " << target_sample_id << " not found in file." << endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]){
    if (argc < 5) {
        cerr << "[HEDim main.cpp] Usage: " << argv[0] << " <path_to_file> <N> <sample_id_a> <sample_id_b>" << endl;
        return EXIT_FAILURE;
    }

    // Gets values from argv
    string path = argv[1];
    uint32_t nslots = static_cast<uint32_t>(stoi(argv[2]));
    int sample_id_a = stoi(argv[3]);
    int sample_id_b = stoi(argv[4]);
    static const uint32_t bitsize = 8; // 8 significant bits, however a sign bit will be used
    static const uint32_t max_bitsize = 24; // the final result will be on 24 bits, in order to ensure correctness of the euclidian distance


    /*
     * Initialisation for the homomorphic encryption
     */
    const int minimum_lambda = 128;
    TFheGateBootstrappingParameterSet* params = new_default_gate_bootstrapping_parameters(minimum_lambda); //The params, public, important for ?? homomorphic operation, in particular generating keys and bootstrapping.
    TFheGateBootstrappingSecretKeySet* key = new_random_gate_bootstrapping_secret_keyset(params);  //The secret key associated with the Client with the particular ID
    const TFheGateBootstrappingCloudKeySet* cloud_key = &key->cloud;

    /*
     * Now gets the value of the vectors
     */
    vector<uint8_t> sample_a = load_sample(path, sample_id_a, nslots);
    vector<uint8_t> sample_b = load_sample(path, sample_id_b, nslots);


    cout << "[HEDim main.cpp] Sample A (ID " << sample_id_a << "):" << endl;
    printVector(sample_a, nslots);
    cout << endl << "[HEDim main.cpp] Sample B (ID " << sample_id_b << "):" << endl;
    printVector(sample_b, nslots);
    cout << endl;

    vector<LweSample*> enc_sample_a(nslots);
    vector<LweSample*> enc_sample_b(nslots);
    // Encryption
    for (int i = 0; i < nslots; i++) {
        enc_sample_a[i] = new_gate_bootstrapping_ciphertext_array(bitsize, params);
        enc_sample_b[i] = new_gate_bootstrapping_ciphertext_array(bitsize, params);
        for (int j=0; j < bitsize; j++) {
            uint8_t mu = sample_a[i];
            bootsSymEncrypt(&enc_sample_a[i][j], (mu>>j)&1, key);
            mu = sample_b[i];
            bootsSymEncrypt(&enc_sample_b[i][j], (mu>>j)&1, key);
        }
    }

    LweSample* tmp = new_gate_bootstrapping_ciphertext_array(max_bitsize, params);
    LweSample* tmp_carry = new_gate_bootstrapping_ciphertext_array(1, cloud_key->params);
    LweSample* tmp_dist = new_gate_bootstrapping_ciphertext_array(max_bitsize, params);
    for (int i = 0; i < max_bitsize; ++i) {
        bootsCONSTANT(&tmp_dist[i], 0, cloud_key);
    }
    auto t_from = Clock::now();
    auto t_to = Clock::now();
    t_from = Clock::now();
    HE_EuclideanDistance(tmp_dist, enc_sample_a, enc_sample_b, bitsize, cloud_key);
    t_to = Clock::now();
    cout << "[HEDim main.cpp] Executed in "
         << chrono::duration_cast<chrono::seconds>(t_to - t_from).count()
         << " seconds" << endl;
    
    uint64_t tmp_result = 0;
    for (int m = 0; m < max_bitsize; ++m) {
        uint8_t dec_mu = bootsSymDecrypt(&tmp_dist[m], key);
        cout << dec_mu << endl;
        tmp_result |= (dec_mu<<m);
    }
    uint64_t eh =  EuclideanDistance(sample_a, sample_b);
    printf("[HEDim main.cpp] The result for HE Euclidean distance is %lu (expected %lu)\n", tmp_result, eh);

    /*
     * Cleaning part
     */
    delete_gate_bootstrapping_ciphertext_array(max_bitsize, tmp);
    delete_gate_bootstrapping_ciphertext_array(1, tmp_carry);
    delete_gate_bootstrapping_ciphertext_array(max_bitsize, tmp_dist);
    delete_gate_bootstrapping_secret_keyset(key);
    delete_gate_bootstrapping_parameters(params);
    for (int i = 0; i < nslots; ++i) {
        delete_gate_bootstrapping_ciphertext_array(bitsize, enc_sample_a[i]);
        delete_gate_bootstrapping_ciphertext_array(bitsize, enc_sample_b[i]);
    }
    return 0;
}