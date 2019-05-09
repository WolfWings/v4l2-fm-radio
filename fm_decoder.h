#ifndef __FM_DECODER_H__
#define __FM_DECODER_H__

#include <stdint.h>

void fm_decoder_process( uint8_t *input, int32_t len, uint16_t *output );
void fm_decoder_reset( void );
void fm_decoder_volume( uint16_t factor ); // 4096 is 100% safe, higher can clip

#endif
