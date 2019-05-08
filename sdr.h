#ifndef __SDR_H__
#define __SDR_H__

#include <stdint.h>

int sdr_tune( int handle, uint32_t freq );
int sdr_init( void );
void sdr_close( void );

#endif
