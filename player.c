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
		0x46464952	// "RIFF"
	,	~0		// near-infinite total size to allow streaming
	,	0x45564157	// "WAVE"
	,	0x20746d66	// "fmt "
	,	16		// size of the header
	,	0x10001		// 1 channel (mono) + PCM format
	,	32000		// Sample rate
	,	64000		// Bytes per second, essentially
	,	0x00100002	// 2-block sample alignment + 16 bits per sample
	,	0x61746164	// "data"
	,	~0		// near-infinite size to allow streaming
	};

	if ( argc < 2 ) {
		fprintf( stderr, "Usage:\n\t%s <frequency> | aplay\n", argv[0] );
		return 1;
	}

	// This construct limits any input number to valid NA FM frequencies only
	dongle.freq = ( uint32_t )( atof( argv[1] ) * 5 );
	dongle.freq = ( dongle.freq + 160 ) % 100;

	dongle.handle = sdr_init();
	if ( dongle.handle == -1 ) {
		fprintf( stderr, "Failed to locate/initialize Software Defined Radio.\n" );
		return 1;
	}

	// The construct reverses the initial compaction of inputs, to avoid needing
	// any real error handling for invalid frequency requests: You just get some
	// valid frequency that is modulus nearby automagically.
	if ( sdr_tune( dongle.handle, ( ( dongle.freq + 440 ) * 200000 ) + 100000 ) != 0 ) {
		fprintf( stderr, "Failed to set radio frequency.\n" );
		close( dongle.handle );
		return 1;
	}

	// Un-required for now, but prepatory for later
	// zero-copy work using this basic codebase.
	input = mmap( NULL, 2048 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0 );
	if ( input == MAP_FAILED ) {
		fprintf( stderr, "Failed to allocate input buffer!\n" );
	}

	// Same as the above, used mmap for one might as well for both.
	output = mmap( NULL, 32768, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0 );
	if ( output == MAP_FAILED ) {
		fprintf( stderr, "Failed to allocate output buffer!\n" );
	}

	// Need to reset the 'carryover' IQ values to avoid a first-sample noise blart
	fm_decoder_reset();

	// Pitch out the WAV header so we can be piped into aplay/play/mplayer/etc
	fwrite( wav_header, sizeof( wav_header ), 1, stdout );

	for ( ; ; ) {
		bytes = read( dongle.handle, input, 2048 * 1024 );
		if ( bytes == -1 ) {
			break;
		}

		// The decoder routine requires 128 bytes
		// for each 2-byte sample generated.
		//
		// In practice this is never an issue as the
		// read gets full buffers or at least power-
		// of-2 worth of data at once.
		bytes &= ~127;

		// Make it so, Number 2!
		fm_decoder_process( input, bytes, output );

		// ...and play me those sweet, dulcet tones.
		fwrite( output, bytes / 64, 1, stdout );
	}

	close( dongle.handle );
}
