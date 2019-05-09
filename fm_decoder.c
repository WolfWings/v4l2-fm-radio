#include <math.h>
#include <stdint.h>
#include <immintrin.h>

#define SIMD_CONST_8(x) { x, x, x, x, x, x, x, x }

static const __v8si fabs_mask		= SIMD_CONST_8(    0x7fffffff ); // Vectored FABS(), just add AND!
static const __v8sf atan2_7		= SIMD_CONST_8( -0.0464964749 ); // a^7 term
static const __v8sf atan2_5		= SIMD_CONST_8(  0.1593142200 ); // a^5 term
static const __v8sf atan2_3		= SIMD_CONST_8( -0.3276227640 ); // a^3 term
static const __v8sf atan2_pi_half	= SIMD_CONST_8( M_PI_2 );
static const __v8sf atan2_pi_full	= SIMD_CONST_8( M_PI );
static __v8sf carryover_i		= SIMD_CONST_8( 0.0 );
static __v8sf carryover_q		= SIMD_CONST_8( 0.0 );
static float volume			= 4096.0 / M_PI;

#define sign_flip_bytes			(__m256i)_sign_flip_bytes
static const __v32qi _sign_flip_bytes	= {-128,-128,-128,-128,-128,-128,-128,-128,
					   -128,-128,-128,-128,-128,-128,-128,-128,
					   -128,-128,-128,-128,-128,-128,-128,-128,
					   -128,-128,-128,-128,-128,-128,-128,-128 };

#define deinterleave_i			(__m256i)_deinterleave_i
static const __v32qi _deinterleave_i	= { 0, 2, 4, 6, 8, 10, 12, 14,
					   -128,-128,-128,-128,-128,-128,-128,-128,
					    0, 2, 4, 6, 8, 10, 12, 14,
					   -128,-128,-128,-128,-128,-128,-128,-128 };

#define deinterleave_q			(__m256i)_deinterleave_q
static const __v32qi _deinterleave_q	= { 1, 3, 5, 7, 9, 11, 13, 15,
					   -128,-128,-128,-128,-128,-128,-128,-128,
					    1, 3, 5, 7, 9, 11, 13, 15,
					   -128,-128,-128,-128,-128,-128,-128,-128 };

#define old_update			(__m256i)_old_update
static const __v8si _old_update		= { 7, 0, 1, 2, 3, 4, 5, 6 };

void fm_decoder_process( uint8_t *raw_input, int32_t len, uint16_t *output ) {
	__v8sf x, y, arctans;
	__m256i raw_0, raw_1, raw_2, raw_3;
	__m256i b_i_0, b_i_1, b_i_2, b_i_3;
	__m256i b_q_0, b_q_1, b_q_2, b_q_3;
	__m256i b_i_0_1, b_i_2_3, b_i_0_3;
	__m256i b_q_0_1, b_q_2_3, b_q_0_3;
	__v8sf new_i, new_q, old_i, old_q;
	__m128i *input = (__m128i*)raw_input;
	__m128i *goal = (__m128i*)(raw_input + len);
	__v4sf final;

	// Pre-load the carryover mechanism
	new_i = carryover_i;
	new_q = carryover_q;

	// The input data structure is 8-bit unsigned IQIQIQIQIQIQIQIQ...
	// * Convert this to signed (byte - 128 = byte ^ 128)
	// * Split into separate I and Q streams
	// * Sign-extend to signed words to retain math overhead
	// * Box-filter decimate 8 samples per stream using horizontal add
	//   As we have sufficient space with the 16-bit capacity we avoid
	//   precision loss as we can skip the divide step
	// * Sign-extend to signed double-words (32-bit)
	// * Bulk-convert to signed single-precision floating point (32-bit)
	do {
		// Carryover terms from last block
		old_i	= new_i;
		old_q	= new_q;

		raw_0	= _mm256_castsi128_si256(         input[0]    ); // Byte offset:   0
		raw_1	= _mm256_castsi128_si256(         input[1]    ); // Byte offset:  16
		raw_2	= _mm256_castsi128_si256(         input[2]    ); // Byte offset:  32
		raw_3	= _mm256_castsi128_si256(         input[3]    ); // Byte offset:  48

		raw_0	= _mm256_insertf128_si256( raw_0, input[4], 1 ); // Byte offset:  64
		raw_1	= _mm256_insertf128_si256( raw_1, input[5], 1 ); // Byte offset:  80
		raw_2	= _mm256_insertf128_si256( raw_2, input[6], 1 ); // Byte offset:  96
		raw_3	= _mm256_insertf128_si256( raw_3, input[7], 1 ); // Byte offset: 112

		input += 8;

		raw_0	= _mm256_xor_si256( raw_0, sign_flip_bytes );
		raw_1	= _mm256_xor_si256( raw_1, sign_flip_bytes );
		raw_2	= _mm256_xor_si256( raw_2, sign_flip_bytes );
		raw_3	= _mm256_xor_si256( raw_3, sign_flip_bytes );

		b_i_0	= _mm256_shuffle_epi8( raw_0, deinterleave_i );
		b_i_1	= _mm256_shuffle_epi8( raw_1, deinterleave_i );
		b_i_2	= _mm256_shuffle_epi8( raw_2, deinterleave_i );
		b_i_3	= _mm256_shuffle_epi8( raw_3, deinterleave_i );

		b_i_0	= _mm256_permute4x64_epi64( b_i_0, _MM_SHUFFLE( 3, 1, 2, 0 ) );
		b_i_1	= _mm256_permute4x64_epi64( b_i_1, _MM_SHUFFLE( 3, 1, 2, 0 ) );
		b_i_2	= _mm256_permute4x64_epi64( b_i_2, _MM_SHUFFLE( 3, 1, 2, 0 ) );
		b_i_3	= _mm256_permute4x64_epi64( b_i_3, _MM_SHUFFLE( 3, 1, 2, 0 ) );

		b_i_0	= _mm256_cvtepi8_epi16( _mm256_castsi256_si128( b_i_0 ) );
		b_i_1	= _mm256_cvtepi8_epi16( _mm256_castsi256_si128( b_i_1 ) );
		b_i_2	= _mm256_cvtepi8_epi16( _mm256_castsi256_si128( b_i_2 ) );
		b_i_3	= _mm256_cvtepi8_epi16( _mm256_castsi256_si128( b_i_3 ) );

		b_i_0_1	= _mm256_hadd_epi16( b_i_0,   b_i_1   );
		b_i_2_3	= _mm256_hadd_epi16( b_i_2,   b_i_3   );
		b_i_0_3	= _mm256_hadd_epi16( b_i_0_1, b_i_2_3 );

		b_q_0	= _mm256_shuffle_epi8( raw_0, deinterleave_q );
		b_q_1	= _mm256_shuffle_epi8( raw_1, deinterleave_q );
		b_q_2	= _mm256_shuffle_epi8( raw_2, deinterleave_q );
		b_q_3	= _mm256_shuffle_epi8( raw_3, deinterleave_q );

		b_q_0	= _mm256_permute4x64_epi64( b_q_0, _MM_SHUFFLE( 3, 1, 2, 0 ) );
		b_q_1	= _mm256_permute4x64_epi64( b_q_1, _MM_SHUFFLE( 3, 1, 2, 0 ) );
		b_q_2	= _mm256_permute4x64_epi64( b_q_2, _MM_SHUFFLE( 3, 1, 2, 0 ) );
		b_q_3	= _mm256_permute4x64_epi64( b_q_3, _MM_SHUFFLE( 3, 1, 2, 0 ) );

		b_q_0	= _mm256_cvtepi8_epi16( _mm256_castsi256_si128( b_q_0 ) );
		b_q_1	= _mm256_cvtepi8_epi16( _mm256_castsi256_si128( b_q_1 ) );
		b_q_2	= _mm256_cvtepi8_epi16( _mm256_castsi256_si128( b_q_2 ) );
		b_q_3	= _mm256_cvtepi8_epi16( _mm256_castsi256_si128( b_q_3 ) );

		b_q_0_1	= _mm256_hadd_epi16( b_q_0,   b_q_1   );
		b_q_2_3	= _mm256_hadd_epi16( b_q_2,   b_q_3   );
		b_q_0_3	= _mm256_hadd_epi16( b_q_0_1, b_q_2_3 );

		b_i_0_3	= _mm256_hadd_epi16( b_i_0_3, b_i_0_3 );
		b_q_0_3	= _mm256_hadd_epi16( b_q_0_3, b_q_0_3 );

		b_i_0_3	= _mm256_permute4x64_epi64( b_i_0_3, _MM_SHUFFLE( 3, 1, 2, 0 ) );
		b_q_0_3	= _mm256_permute4x64_epi64( b_q_0_3, _MM_SHUFFLE( 3, 1, 2, 0 ) );

		b_i_0_3	= _mm256_cvtepi16_epi32( _mm256_castsi256_si128( b_i_0_3 ) );
		b_q_0_3	= _mm256_cvtepi16_epi32( _mm256_castsi256_si128( b_q_0_3 ) );

		new_i	= _mm256_cvtepi32_ps( b_i_0_3 );
		new_q	= _mm256_cvtepi32_ps( b_q_0_3 );

		old_i	= _mm256_blend_ps( new_i, old_i, 0x80 );
		old_q	= _mm256_blend_ps( new_q, old_q, 0x80 );

		old_i	= _mm256_permutevar8x32_ps( old_i, old_update );
		old_q	= _mm256_permutevar8x32_ps( old_q, old_update );

		y = ( old_i * new_q ) - ( old_q * new_i );
		x = ( old_i * new_i ) + ( old_q * new_q );

		// Basic concept here is built around the equation of:
		//	min( abs( x ), abs( y ) ) / max( abs( x ), abs( y ) )
		// This maps the entire circle into a single 1/8th wedge
		// and from there a limited 4 term 7th degree polynomial
		// works and we map it's output to the other 8 wedges

		__v8sf yabs = _mm256_and_ps( y, (__v8sf)fabs_mask );
		__v8sf xabs = _mm256_and_ps( x, (__v8sf)fabs_mask );
		__v8sf maxv = _mm256_max_ps( xabs, yabs );
		__v8sf minv = _mm256_min_ps( xabs, yabs );
		__v8sf div = minv / maxv;
		__v8sf sqr = div * div;
		arctans = ( ( ( ( ( atan2_7 * sqr ) + atan2_5 ) * sqr ) + atan2_3 ) * div * sqr ) + div;

		// Handle x/y versus y/x inversion + rotation
		__v8sf flip_mask = _mm256_cmp_ps( yabs, xabs, _CMP_GT_OQ );
		__v8sf flip_value = _mm256_sub_ps( atan2_pi_half, arctans );
		arctans = _mm256_blendv_ps( arctans, flip_value, flip_mask );

		// Negative X inversion + rotation
		// As x86 SIMD requires the highest bit to indicate sign
		// and the BlendV operation only checks the top bit,
		// we can avoid an extra register used here by simply
		// referencing X directly as the 'mask' register
		__v8sf negx_value = _mm256_sub_ps( atan2_pi_full, arctans );
		arctans = _mm256_blendv_ps( arctans, negx_value, x );

		// Negative Y inversion
		// As we are strictly negating arctans if the matching
		// entry in Y is negative, we can simply xor in Y's
		// sign bits here.
		// To extract Y's sign bits we'll cheat and xor abs(y)
		// back in since we only masked the sign bits out to
		// create the abs(y) value earlier.
		arctans = _mm256_xor_ps( arctans, _mm256_xor_ps( y, yabs ) );

		// At this point our actual output audio sample is box-filtered
		// from 8 samples to 1, and converted from radians to a signed
		// 16-bit integer.
		//
		// Depending on signal a larger value is safe here, it also is
		// essentially the 'volume' of the output as well.
		arctans = _mm256_hadd_ps( arctans, arctans );
		arctans = _mm256_hadd_ps( arctans, arctans );
		final = _mm_add_ps( _mm256_castps256_ps128( arctans ), _mm256_extractf128_ps( arctans, 1 ) );
		*output++ = (int16_t)(final[0] * volume);

	} while ( input < goal );

	// Save the last pair of samples for carryover on the next buffer
	carryover_i = new_i;
	carryover_q = new_q;
}

void fm_decoder_reset( void ) {
	carryover_i[7] = carryover_q[7] = 0.0;
}

void fm_decoder_volume( uint16_t factor ) {
	volume = ((float)factor) / M_PI;
}
