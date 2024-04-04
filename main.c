#include <stdint.h>
#include <immintrin.h>

static _Alignas(64) uint8_t bitmask[182] = {
        0b10000010, 0b00100100, 0b10001000, 0b10010000, 0b00000001, 0b00010000, 0b00000010, 0b00000000, 0b01010000,
        0b10000000, 0b01000100, 0b01001000, 0b10010010, 0b00100000, 0b00000000, 0b00000000, 0b00100110, 0b00001000,
        0, // seed only needs to consider up to 162
};

const static _Alignas(64) uint32_t sums_of_squares[100] = {
        0 + 0, 0 + 1, 0 + 4, 0 + 9, 0 + 16, 0 + 25, 0 + 36, 0 + 49, 0 + 64, 0 + 81,
        1 + 0, 1 + 1, 1 + 4, 1 + 9, 1 + 16, 1 + 25, 1 + 36, 1 + 49, 1 + 64, 1 + 81,
        4 + 0, 4 + 1, 4 + 4, 4 + 9, 4 + 16, 4 + 25, 4 + 36, 4 + 49, 4 + 64, 4 + 81,
        9 + 0, 9 + 1, 9 + 4, 9 + 9, 9 + 16, 9 + 25, 9 + 36, 9 + 49, 9 + 64, 9 + 81,
        16 + 0, 16 + 1, 16 + 4, 16 + 9, 16 + 16, 16 + 25, 16 + 36, 16 + 49, 16 + 64, 16 + 81,
        25 + 0, 25 + 1, 25 + 4, 25 + 9, 25 + 16, 25 + 25, 25 + 36, 25 + 49, 25 + 64, 25 + 81,
        36 + 0, 36 + 1, 36 + 4, 36 + 9, 36 + 16, 36 + 25, 36 + 36, 36 + 49, 36 + 64, 36 + 81,
        49 + 0, 49 + 1, 49 + 4, 49 + 9, 49 + 16, 49 + 25, 49 + 36, 49 + 49, 49 + 64, 49 + 81,
        64 + 0, 64 + 1, 64 + 4, 64 + 9, 64 + 16, 64 + 25, 64 + 36, 64 + 49, 64 + 64, 64 + 81,
        81 + 0, 81 + 1, 81 + 4, 81 + 9, 81 + 16, 81 + 25, 81 + 36, 81 + 49, 81 + 64, 81 + 81
};

const static _Alignas(64) uint32_t digits[100] = {
        0x3030, 0x3130, 0x3230, 0x3330, 0x3430, 0x3530, 0x3630, 0x3730, 0x3830, 0x3930,
        0x3031, 0x3131, 0x3231, 0x3331, 0x3431, 0x3531, 0x3631, 0x3731, 0x3831, 0x3931,
        0x3032, 0x3132, 0x3232, 0x3332, 0x3432, 0x3532, 0x3632, 0x3732, 0x3832, 0x3932,
        0x3033, 0x3133, 0x3233, 0x3333, 0x3433, 0x3533, 0x3633, 0x3733, 0x3833, 0x3933,
        0x3034, 0x3134, 0x3234, 0x3334, 0x3434, 0x3534, 0x3634, 0x3734, 0x3834, 0x3934,
        0x3035, 0x3135, 0x3235, 0x3335, 0x3435, 0x3535, 0x3635, 0x3735, 0x3835, 0x3935,
        0x3036, 0x3136, 0x3236, 0x3336, 0x3436, 0x3536, 0x3636, 0x3736, 0x3836, 0x3936,
        0x3037, 0x3137, 0x3237, 0x3337, 0x3437, 0x3537, 0x3637, 0x3737, 0x3837, 0x3937,
        0x3038, 0x3138, 0x3238, 0x3338, 0x3438, 0x3538, 0x3638, 0x3738, 0x3838, 0x3938,
        0x3039, 0x3139, 0x3239, 0x3339, 0x3439, 0x3539, 0x3639, 0x3739, 0x3839, 0x3939,
};

const static uint64_t one_million = 0x1000000; // We are in BCD!
const static uint64_t ninetynines = 0x63636363;
const static uint64_t minusninetynines = 0x9D9D9D9D;
const static uint64_t minusones = 0xFFFFFFFFFFFFFFFF;
const static _Alignas(16) uint64_t shuffleidxs[2] = { 0x0100050409080D0C, 0xFFFFFFFFFFFFFFFF };
const static uint64_t zeroes = 0x3030303030303030;

#include <unistd.h>

int main() {
    const __m128i millionvec = _mm_set_epi64x(0, one_million);
    const __m128i ninesvec = _mm_set_epi64x(0, ninetynines);
    const __m128i minusonesvec = _mm_set_epi64x(0, minusones);
    const __m128i shuffleidxsvec = _mm_set_epi64x(shuffleidxs[1], shuffleidxs[0]);
    const __m128i zeroesvec = _mm_set_epi64x(0, zeroes);

    __m128i accumulator = _mm_setzero_si128();

    while(true) {
        __m128i flagNinesAsOnes = _mm_cmpeq_epi8(ninesvec, accumulator);
        uint64_t nonNineBytes = _mm_cvtsi128_si64(flagNinesAsOnes);
        // takes nines 0xFF to 0x00, takes everything else 0x00 to 0xFF
        uint64_t nineBytes = ~nonNineBytes;

        uint64_t firstNonNineBit = _tzcnt_u64(nineBytes);
        uint64_t setSubtractions = minusninetynines & ((1 << firstNonNineBit) - 1);
        uint64_t setAdditions = 1 << firstNonNineBit;
        uint64_t total = setSubtractions | setAdditions;

        accumulator = _mm_add_epi8(accumulator, _mm_set_epi64x(0, total));

        __m128i fourBytesToFourWords = _mm_unpacklo_epi8(accumulator, _mm_setzero_si128());
        __m128i fourDwords = _mm_unpacklo_epi8(fourBytesToFourWords, _mm_setzero_si128());
        __m128i sumOfSquares = _mm_i32gather_epi32(&sums_of_squares[0], fourDwords, 4);
        uint64_t idx = _mm_extract_epi32(sumOfSquares, 0) + _mm_extract_epi32(sumOfSquares, 1) +
                _mm_extract_epi32(sumOfSquares, 2) + _mm_extract_epi32(sumOfSquares, 3);

        uint64_t bitPull = 1 << (idx & 7);

        if(bitmask[idx >> 3] & bitPull) {
            uint64_t storeIdx = _mm_extract_epi8(accumulator, 0) + (100 * _mm_extract_epi8(accumulator, 1)) +
                    (10000 * _mm_extract_epi8(accumulator, 2)) + (1000000 * _mm_extract_epi8(accumulator, 3));
            if((storeIdx >> 3) < 182) {
                uint64_t storeBitPull = 1 << (storeIdx & 7);
                bitmask[storeIdx >> 3] |= storeBitPull;
            }

            __m128i charSeq16s = _mm_i32gather_epi32(&digits[0], fourDwords, 4);
            __m128i permChars = _mm_shuffle_epi8(charSeq16s, shuffleidxsvec);
            const char actualChars[10] = {
                    0, 0, 0, 0, 0, 0, 0, 0,
                    '\n'
            };
            _mm_maskmoveu_si128(permChars, minusonesvec, (void*) &actualChars[0]);
            int charsWritten = _mm_cmpistri(permChars, zeroesvec, 0b00011000);
            do {
                charsWritten +=
                        write(STDOUT_FILENO, &actualChars[charsWritten], 9 - charsWritten);
            } while(__builtin_expect(charsWritten, 9) < 9);
        }

        __m128i shouldExit = _mm_cmpeq_epi32(accumulator, millionvec);
        uint32_t result = _mm_cvtsi128_si32(shouldExit);
        if(__builtin_expect(result != 0, 0)) return 0;
    }
}
