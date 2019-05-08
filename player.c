#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/mman.h>

#include "fm_decoder.h"
#include "sdr.h"

struct {
	int			handle;
	uint32_t		freq;
} dongle;

int main( int argc, char **argv ) {
	uint8_t *input;
	uint16_t *output;
	ssize_t bytes;

	// Allows using aplay without tons of options
	uint32_t wav_header[] = {
		0x46464952
	,	~0
	,	0x45564157
	,	0x20746d66
	,	16
	,	0x10001
	,	32000
	,	64000
	,	0x00100002
	,	0x61746164
	,	~0
	};

	if ( argc < 2 ) {
		fprintf( stderr, "Usage:\n\t%s <frequency>\n", argv[0] );
		return 1;
	}

	dongle.freq = ( uint32_t )( atof( argv[1] ) * 5 );
	dongle.freq = ( dongle.freq + 160 ) % 100;

	dongle.handle = sdr_init();
	if ( dongle.handle == -1 ) {
		fprintf( stderr, "Failed to locate Software Defined Radio.\n" );
		return 1;
	}

	if ( sdr_tune( dongle.handle, ( ( dongle.freq + 440 ) * 200000 ) + 100000 ) != 0 ) {
		fprintf( stderr, "Failed to set radio frequency.\n" );
		sdr_close();
	}

	input = mmap( NULL, 2048 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0 );
	if ( input == MAP_FAILED ) {
		fprintf( stderr, "Failed to allocate input buffer!\n" );
	}

	output = mmap( NULL, 32768, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0 );
	if ( output == MAP_FAILED ) {
		fprintf( stderr, "Failed to allocate output buffer!\n" );
	}

	fm_decoder_reset();

	fwrite( wav_header, sizeof( wav_header ), 1, stdout );
	for ( ; ; ) {
		bytes = read( dongle.handle, input, 2048 * 1024 );
		if ( bytes == -1 ) {
			break;
		}
		bytes &= ~63;
		fm_decoder_process( input, bytes, output );
		fwrite( output, bytes / 64, 1, stdout );
	}

	sdr_close();
}
