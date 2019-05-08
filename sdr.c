#include <fcntl.h>
#include <dirent.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <linux/v4l2-subdev.h>

struct {
	struct v4l2_tuner sampler;
	struct v4l2_tuner tuner;
	int handle;
} sdr;

static struct v4l2_frequency scratch_freq;

int sdr_tune( int handle, uint32_t freq ) {
	memset( &scratch_freq, 0, sizeof( scratch_freq ) );
	scratch_freq.type = V4L2_TUNER_RF;
	scratch_freq.tuner = 1;
	scratch_freq.frequency = freq;
	if ( ( ioctl( handle, VIDIOC_G_FREQUENCY, &scratch_freq ) == 0 )
	  && ( scratch_freq.frequency == freq ) ) {
		return 0;
	}

	memset( &scratch_freq, 0, sizeof( scratch_freq ) );
	scratch_freq.type = V4L2_TUNER_RF;
	scratch_freq.tuner = 1;
	scratch_freq.frequency = freq;

	return ioctl( handle, VIDIOC_S_FREQUENCY, &scratch_freq );
}

int sdr_init( void ) {
	struct dirent *entry;
	DIR *devices = opendir( "/dev" );

	sdr.handle = -1;

	for ( ; ; ) {
		if ( sdr.handle != -1 ) {
			close( sdr.handle );
			sdr.handle = -1;
		}

		entry = readdir( devices );

		if ( entry == NULL ) {
			break;
		}

		if ( entry->d_type != DT_CHR ) {
			continue;
		}

		if ( strncmp( entry->d_name, "swradio", 7 ) != 0 ) {
			continue;
		}

		sdr.handle = openat( dirfd( devices ), entry->d_name, O_RDONLY );

		if ( sdr.handle == -1 ) {
			continue;
		}

		memset( &sdr.sampler, 0, sizeof( struct v4l2_tuner ) );
		sdr.sampler.index = 0;

		if ( ioctl( sdr.handle, VIDIOC_G_TUNER, &sdr.sampler ) == -1 ) {
			continue;
		}

		// The decoder is built around a fixed 2.048MBit -> 32KBit mechanism
		if ( ( sdr.sampler.type != V4L2_TUNER_SDR )
		  || ( sdr.sampler.rangelow > 2048000 )
		  || ( sdr.sampler.rangehigh < 2048000 ) ) {
			continue;
		}

		memset( &sdr.tuner,   0, sizeof( struct v4l2_tuner ) );
		sdr.tuner.index = 1;

		if ( ioctl( sdr.handle, VIDIOC_G_TUNER, &sdr.tuner ) == -1 ) {
			continue;
		}

		// Make sure we can actually cover the FM radio band, our main goal
		if ( ( sdr.tuner.type != V4L2_TUNER_RF )
		  || ( ( sdr.tuner.capability & V4L2_TUNER_CAP_1HZ ) == 0 )
		  || ( sdr.tuner.rangelow  >  88100000 )
		  || ( sdr.tuner.rangehigh < 107900000 ) ) {
			continue;
		}

		// Reduce restart time by only setting the sampling rate if it's not already 2048000
		memset( &scratch_freq, 0, sizeof( scratch_freq ) );
		scratch_freq.type = V4L2_TUNER_SDR;
		scratch_freq.tuner = 0;

		if ( ioctl( sdr.handle, VIDIOC_G_FREQUENCY, &scratch_freq ) != 0 ) {
			continue;
		}

		if ( scratch_freq.frequency != 2048000 ) {
			memset( &scratch_freq, 0, sizeof( scratch_freq ) );
			scratch_freq.type = V4L2_TUNER_SDR;
			scratch_freq.tuner = 0;
			scratch_freq.frequency = 2048000;

			if ( ioctl( sdr.handle, VIDIOC_S_FREQUENCY, &scratch_freq ) != 0 ) {
				continue;
			}
		}

		break;
	}

	closedir( devices );

	return sdr.handle;
}

void sdr_close( void ) {
	if ( sdr.handle != -1 ) {
		close( sdr.handle );
		sdr.handle = -1;
	}
}
