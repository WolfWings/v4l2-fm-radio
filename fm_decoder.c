#include <math.h>
#include <stdint.h>
#include <immintrin.h>

#define SIMD_CONST_8(x) { x, x, x, x, x, x, x, x }

static const __v8si fabs_mask		= SIMD_CONST_8( 0x7fffffff );
static const __v8sf atan2_2		= SIMD_CONST_8( -0.0464964749 );
static const __v8sf atan2_1		= SIMD_CONST_8(  0.1593142200 );
static const __v8sf atan2_0		= SIMD_CONST_8( -0.3276227640 );
static const __v8sf atan2_pi_half	= SIMD_CONST_8( M_PI_2 );
static const __v8sf atan2_pi_full	= SIMD_CONST_8( M_PI );
static const __v8sf zero		= SIMD_CONST_8( 0.0 );
static const __v8hi sign_flip_bytes	= SIMD_CONST_8( ~0x7f7f );
static const __v16qi deinterleave_i	= { 0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128 };
static const __v16qi deinterleave_q	= { 1, 3, 5, 7, 9, 11, 13, 15, -128, -128, -128, -128, -128, -128, -128, -128 };

/* Moved outside of the processing function to avoid being stack allocated */
static float mux_i[16] __attribute__((aligned (16)));
static float mux_q[16] __attribute__((aligned (16)));

static float carryover_i;
static float carryover_q;

void fm_decoder_process( uint8_t *input, int32_t len, uint16_t *output ) {
	__v8sf x, y, arctans;
	__m128i b_i_0, b_i_1, b_i_2, b_i_3;
	__m128i b_q_0, b_q_1, b_q_2, b_q_3;

	mux_i[8] = carryover_i;
	mux_q[8] = carryover_q;

	for ( uint8_t *goal = input + len; goal > input; input += 128 ) {
		mux_i[0] = mux_i[8];
		mux_q[0] = mux_q[8];

		b_q_0	= _mm_xor_si128( *(__m128i*)(&input[  0]), (__m128i)sign_flip_bytes );
		b_i_0	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_0, (__m128i)deinterleave_i ) );
		b_q_0	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_0, (__m128i)deinterleave_q ) );

		b_q_1	= _mm_xor_si128( *(__m128i*)(&input[ 16]), (__m128i)sign_flip_bytes );
		b_i_1	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_1, (__m128i)deinterleave_i ) );
		b_q_1	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_1, (__m128i)deinterleave_q ) );

		b_i_0	= _mm_hadd_epi16( b_i_0, b_i_1 );
		b_q_0	= _mm_hadd_epi16( b_q_0, b_q_1 );

		b_q_2	= _mm_xor_si128( *(__m128i*)(&input[ 32]), (__m128i)sign_flip_bytes );
		b_i_2	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_2, (__m128i)deinterleave_i ) );
		b_q_2	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_2, (__m128i)deinterleave_q ) );

		b_q_3	= _mm_xor_si128( *(__m128i*)(&input[ 48]), (__m128i)sign_flip_bytes );
		b_i_3	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_3, (__m128i)deinterleave_i ) );
		b_q_3	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_3, (__m128i)deinterleave_q ) );

		b_i_2	= _mm_hadd_epi16( b_i_2, b_i_3 );
		b_q_2	= _mm_hadd_epi16( b_q_2, b_q_3 );

		b_i_0	= _mm_hadd_epi16( b_i_0, b_i_2 );
		b_q_0	= _mm_hadd_epi16( b_q_0, b_q_2 );

		b_i_0	= _mm_hadd_epi16( b_i_0, b_i_0 );
		b_q_0	= _mm_hadd_epi16( b_q_0, b_q_0 );

		b_i_0	= _mm_cvtepi16_epi32( b_i_0 );
		b_q_0	= _mm_cvtepi16_epi32( b_q_0 );

		*(__m128*)(&mux_i[1]) = _mm_cvtepi32_ps( b_i_0 );
		*(__m128*)(&mux_q[1]) = _mm_cvtepi32_ps( b_q_0 );

		b_q_0	= _mm_xor_si128( *(__m128i*)(&input[ 64]), (__m128i)sign_flip_bytes );
		b_i_0	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_0, (__m128i)deinterleave_i ) );
		b_q_0	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_0, (__m128i)deinterleave_q ) );

		b_q_1	= _mm_xor_si128( *(__m128i*)(&input[ 80]), (__m128i)sign_flip_bytes );
		b_i_1	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_1, (__m128i)deinterleave_i ) );
		b_q_1	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_1, (__m128i)deinterleave_q ) );

		b_i_0	= _mm_hadd_epi16( b_i_0, b_i_1 );
		b_q_0	= _mm_hadd_epi16( b_q_0, b_q_1 );

		b_q_2	= _mm_xor_si128( *(__m128i*)(&input[ 96]), (__m128i)sign_flip_bytes );
		b_i_2	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_2, (__m128i)deinterleave_i ) );
		b_q_2	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_2, (__m128i)deinterleave_q ) );

		b_q_3	= _mm_xor_si128( *(__m128i*)(&input[112]), (__m128i)sign_flip_bytes );
		b_i_3	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_3, (__m128i)deinterleave_i ) );
		b_q_3	= _mm_cvtepi8_epi16( _mm_shuffle_epi8( b_q_3, (__m128i)deinterleave_q ) );

		b_i_2	= _mm_hadd_epi16( b_i_2, b_i_3 );
		b_q_2	= _mm_hadd_epi16( b_q_2, b_q_3 );

		b_i_0	= _mm_hadd_epi16( b_i_0, b_i_2 );
		b_q_0	= _mm_hadd_epi16( b_q_0, b_q_2 );

		b_i_0	= _mm_hadd_epi16( b_i_0, b_i_0 );
		b_q_0	= _mm_hadd_epi16( b_q_0, b_q_0 );

		b_i_0	= _mm_cvtepi16_epi32( b_i_0 );
		b_q_0	= _mm_cvtepi16_epi32( b_q_0 );

		*(__m128*)(&mux_i[5]) = _mm_cvtepi32_ps( b_i_0 );
		*(__m128*)(&mux_q[5]) = _mm_cvtepi32_ps( b_q_0 );

		y = _mm256_sub_ps(
			_mm256_mul_ps( *(__m256*)(&mux_i[0]), *(__m256*)(&mux_q[1]) )
		,	_mm256_mul_ps( *(__m256*)(&mux_i[1]), *(__m256*)(&mux_q[0]) )
		);

		x = _mm256_add_ps(
			_mm256_mul_ps( *(__m256*)(&mux_i[0]), *(__m256*)(&mux_i[1]) )
		,	_mm256_mul_ps( *(__m256*)(&mux_q[0]), *(__m256*)(&mux_q[1]) )
		);

		__v8sf yabs = _mm256_and_ps( y, (__v8sf)fabs_mask );
		__v8sf xabs = _mm256_and_ps( x, (__v8sf)fabs_mask );
		__v8sf maxv = _mm256_max_ps( xabs, yabs );
		__v8sf minv = _mm256_min_ps( xabs, yabs );
		__v8sf div = _mm256_div_ps( minv, maxv );
		__v8sf sqr = _mm256_mul_ps( div, div );
		arctans = _mm256_fmadd_ps( sqr, atan2_2, atan2_1 );
		arctans = _mm256_fmadd_ps( arctans, sqr, atan2_0 );
		arctans = _mm256_mul_ps( arctans, sqr );
		arctans = _mm256_fmadd_ps( arctans, div, div );
		__v8sf flip_mask = _mm256_cmp_ps( yabs, xabs, _CMP_GE_OQ );
		__v8sf flip_value = _mm256_sub_ps( atan2_pi_half, arctans );
		arctans = _mm256_blendv_ps( arctans, flip_value, flip_mask );
		__v8sf negx_mask = _mm256_cmp_ps( x, zero, _CMP_LT_OQ );
		__v8sf negx_value = _mm256_sub_ps( atan2_pi_full, arctans );
		arctans = _mm256_blendv_ps( arctans, negx_value, negx_mask );
		__v8sf negy_mask = _mm256_cmp_ps( y, zero, _CMP_LT_OQ );
		__v8sf negy_value = _mm256_sub_ps( zero, arctans );
		arctans = _mm256_blendv_ps( arctans, negy_value, negy_mask );
		arctans = _mm256_hadd_ps( arctans, arctans );
		arctans = _mm256_hadd_ps( arctans, arctans );
		*output++ = (int16_t)( ( arctans[0] + arctans[4] ) * ( 4096.0 / M_PI ) );
	}

	carryover_i = mux_i[8];
	carryover_q = mux_q[8];
}

void fm_decoder_reset( void ) {
	carryover_i = carryover_q = 0.0;
}
