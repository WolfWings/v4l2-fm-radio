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
static const __v8sf zero		= SIMD_CONST_8( 0.0 );
static const __v8hi sign_flip_bytes	= SIMD_CONST_8( ~0x7f7f ); // 16 bytes of 0x80 defined in a way to avoid compiler warnings
static const __v16qi deinterleave_i	= { 0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128 };
static const __v16qi deinterleave_q	= { 1, 3, 5, 7, 9, 11, 13, 15, -128, -128, -128, -128, -128, -128, -128, -128 };

// Moved outside of the processing function to avoid being stack allocated
// Also these are larger than strictly needed to allow for aligned stores
static float mux_i[16] __attribute__((aligned (16)));
static float mux_q[16] __attribute__((aligned (16)));

static float carryover_i;
static float carryover_q;

void fm_decoder_process( uint8_t *input, int32_t len, uint16_t *output ) {
	__v8sf x, y, arctans;
	__m128i b_i_0, b_i_1, b_i_2, b_i_3;
	__m128i b_q_0, b_q_1, b_q_2, b_q_3;

	// Pre-load the carryover mechanism
	mux_i[11] = carryover_i;
	mux_q[11] = carryover_q;

	// The input data structure is 8-bit unsigned IQIQIQIQIQIQIQIQ...
	// * Convert this to signed (byte - 128 = byte ^ 128)
	// * Split into separate I and Q streams
	// * Sign-extend to signed words to retain math overhead
	// * Box-filter decimate 8 samples per stream using horizontal add
	//   As we have sufficient space with the 16-bit capacity we avoid
	//   precision loss as we can skip the divide step
	// * Sign-extend to signed double-words (32-bit)
	// * Bulk-convert to signed single-precision floating point (32-bit)
	for ( uint8_t *goal = input + len; goal > input; input += 128 ) {
		// Carryover terms from last block
		mux_i[3] = mux_i[11];
		mux_q[3] = mux_q[11];

		b_q_0	= _mm_xor_si128( *(__m128i*)(&input[  0]), (__m128i)sign_flip_bytes );
		b_i_0	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_0, (__m128i)deinterleave_i ) );
		b_q_0	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_0, (__m128i)deinterleave_q ) );

		b_q_1	= _mm_xor_si128( *(__m128i*)(&input[ 16]), (__m128i)sign_flip_bytes );
		b_i_1	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_1, (__m128i)deinterleave_i ) );
		b_q_1	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_1, (__m128i)deinterleave_q ) );

		b_q_2	= _mm_xor_si128( *(__m128i*)(&input[ 32]), (__m128i)sign_flip_bytes );
		b_i_2	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_2, (__m128i)deinterleave_i ) );
		b_q_2	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_2, (__m128i)deinterleave_q ) );

		b_q_3	= _mm_xor_si128( *(__m128i*)(&input[ 48]), (__m128i)sign_flip_bytes );
		b_i_3	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_3, (__m128i)deinterleave_i ) );
		b_q_3	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_3, (__m128i)deinterleave_q ) );

		//	I stream data					Q stream data
		b_i_0	= _mm_hadd_epi16( b_i_0, b_i_1 );	b_q_0	= _mm_hadd_epi16( b_q_0, b_q_1 );
		b_i_2	= _mm_hadd_epi16( b_i_2, b_i_3 );	b_q_2	= _mm_hadd_epi16( b_q_2, b_q_3 );
		b_i_0	= _mm_hadd_epi16( b_i_0, b_i_2 );	b_q_0	= _mm_hadd_epi16( b_q_0, b_q_2 );
		b_i_0	= _mm_hadd_epi16( b_i_0, b_i_0 );	b_q_0	= _mm_hadd_epi16( b_q_0, b_q_0 );
		b_i_0	= _mm_cvtepi16_epi32( b_i_0 );		b_q_0	= _mm_cvtepi16_epi32( b_q_0 );

		*(__m128*)(&mux_i[4]) = _mm_cvtepi32_ps( b_i_0 );
		*(__m128*)(&mux_q[4]) = _mm_cvtepi32_ps( b_q_0 );

		b_q_0	= _mm_xor_si128( *(__m128i*)(&input[ 64]), (__m128i)sign_flip_bytes );
		b_i_0	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_0, (__m128i)deinterleave_i ) );
		b_q_0	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_0, (__m128i)deinterleave_q ) );

		b_q_1	= _mm_xor_si128( *(__m128i*)(&input[ 80]), (__m128i)sign_flip_bytes );
		b_i_1	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_1, (__m128i)deinterleave_i ) );
		b_q_1	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_1, (__m128i)deinterleave_q ) );

		b_q_2	= _mm_xor_si128( *(__m128i*)(&input[ 96]), (__m128i)sign_flip_bytes );
		b_i_2	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_2, (__m128i)deinterleave_i ) );
		b_q_2	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_2, (__m128i)deinterleave_q ) );

		b_q_3	= _mm_xor_si128( *(__m128i*)(&input[112]), (__m128i)sign_flip_bytes );
		b_i_3	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_3, (__m128i)deinterleave_i ) );
		b_q_3	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_3, (__m128i)deinterleave_q ) );

		//	I stream data					Q stream data
		b_i_0	= _mm_hadd_epi16( b_i_0, b_i_1 );	b_q_0	= _mm_hadd_epi16( b_q_0, b_q_1 );
		b_i_2	= _mm_hadd_epi16( b_i_2, b_i_3 );	b_q_2	= _mm_hadd_epi16( b_q_2, b_q_3 );
		b_i_0	= _mm_hadd_epi16( b_i_0, b_i_2 );	b_q_0	= _mm_hadd_epi16( b_q_0, b_q_2 );
		b_i_0	= _mm_hadd_epi16( b_i_0, b_i_0 );	b_q_0	= _mm_hadd_epi16( b_q_0, b_q_0 );
		b_i_0	= _mm_cvtepi16_epi32( b_i_0 );		b_q_0	= _mm_cvtepi16_epi32( b_q_0 );

		*(__m128*)(&mux_i[8]) = _mm_cvtepi32_ps( b_i_0 );
		*(__m128*)(&mux_q[8]) = _mm_cvtepi32_ps( b_q_0 );

		// y = i[0] * q[1] - i[1] * q[0]
		y = _mm256_sub_ps(
			_mm256_mul_ps( *(__m256*)(&mux_i[3]), *(__m256*)(&mux_q[4]) )
		,	_mm256_mul_ps( *(__m256*)(&mux_i[4]), *(__m256*)(&mux_q[3]) )
		);

		// x = i[0] * i[1] + q[0] * q[1]
		x = _mm256_add_ps(
			_mm256_mul_ps( *(__m256*)(&mux_i[3]), *(__m256*)(&mux_i[4]) )
		,	_mm256_mul_ps( *(__m256*)(&mux_q[3]), *(__m256*)(&mux_q[4]) )
		);

		// Basic concept here is built around the equation of:
		//	min( abs( x ), abs( y ) ) / max( abs( x ), abs( y ) )
		// This maps the entire circle into a single 1/8th wedge
		// and from there a limited 4 term 7th degree polynomial
		// works and we map it's output to the other 8 wedges

		__v8sf yabs = _mm256_and_ps( y, (__v8sf)fabs_mask );
		__v8sf xabs = _mm256_and_ps( x, (__v8sf)fabs_mask );
		__v8sf maxv = _mm256_max_ps( xabs, yabs );
		__v8sf minv = _mm256_min_ps( xabs, yabs );
		__v8sf div = _mm256_div_ps( minv, maxv );
		__v8sf sqr = _mm256_mul_ps( div, div );
		arctans = _mm256_fmadd_ps( sqr, atan2_7, atan2_5 );
		arctans = _mm256_fmadd_ps( arctans, sqr, atan2_3 );
		arctans = _mm256_mul_ps( arctans, sqr );
		arctans = _mm256_fmadd_ps( arctans, div, div );

		// Handle x/y versus y/x inversion + rotation
		__v8sf flip_mask = _mm256_cmp_ps( yabs, xabs, _CMP_GE_OQ );
		__v8sf flip_value = _mm256_sub_ps( atan2_pi_half, arctans );
		arctans = _mm256_blendv_ps( arctans, flip_value, flip_mask );

		// Negative X inversion + rotation
		__v8sf negx_mask = _mm256_cmp_ps( x, zero, _CMP_LT_OQ );
		__v8sf negx_value = _mm256_sub_ps( atan2_pi_full, arctans );
		arctans = _mm256_blendv_ps( arctans, negx_value, negx_mask );

		// Negative Y inversion
		__v8sf negy_mask = _mm256_cmp_ps( y, zero, _CMP_LT_OQ );
		__v8sf negy_value = _mm256_sub_ps( zero, arctans );
		arctans = _mm256_blendv_ps( arctans, negy_value, negy_mask );

		// At this point our actual output audio sample is box-filtered
		// from 8 samples to 1, and converted from radians to a signed
		// 16-bit integer.
		//
		// Depending on signal a larger value is safe here, it also is
		// essentially the 'volume' of the output as well.
		arctans = _mm256_hadd_ps( arctans, arctans );
		arctans = _mm256_hadd_ps( arctans, arctans );
		*output++ = (int16_t)( ( arctans[0] + arctans[4] ) * ( 4096.0 / M_PI ) );
	}

	// Save the last pair of samples for carryover on the next buffer
	carryover_i = mux_i[11];
	carryover_q = mux_q[11];
}

void fm_decoder_reset( void ) {
	carryover_i = carryover_q = 0.0;
}
