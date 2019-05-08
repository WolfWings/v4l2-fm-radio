#include <fcntl.h>
#include <dirent.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <linux/v4l2-subdev.h>

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
	struct v4l2_tuner sampler;
	struct v4l2_tuner tuner;
	struct dirent *entry;
	DIR *devices = opendir( "/dev" );
	int handle = -1;

	for ( ; ; ) {
		// This is a loop simplification:
		// We don't need to manually close the handle
		// for the device before giving up on an option.
		if ( handle != -1 ) {
			close( handle );
			handle = -1;
		}

		entry = readdir( devices );

		// Out of devices to examine, too bad. :(
		if ( entry == NULL ) {
			break;
		}

		// Early bail-out to ignore anything not a 'character device' node
		if ( entry->d_type != DT_CHR ) {
			continue;
		}

		// Next up check if the device name begins as expected
		if ( strncmp( entry->d_name, "swradio", 7 ) != 0 ) {
			continue;
		}

		handle = openat( dirfd( devices ), entry->d_name, O_RDONLY );

		// Well THAT was anti-climatic, NEXT!
		if ( handle == -1 ) {
			continue;
		}

		// The sampler is always tuner index 0
		memset( &sampler, 0, sizeof( struct v4l2_tuner ) );
		sampler.index = 0;

		// If the ioctl fails, either:
		// * It's not an SDR.
		// * There's a hardware issue.
		//
		// Either way, we don't care, next contestant!
		if ( ioctl( handle, VIDIOC_G_TUNER, &sampler ) == -1 ) {
			continue;
		}

		// The decoder is built around a fixed 2.048MBit -> 32KBit function
		// so verify the sampler can (nominally) support that range.
		//
		// Note that this does NOT do a comprehensive test against the
		// disparate range list the API can return. This is an early-out
		// for rapid auto-detection not intended for debugging hardware.
		if ( ( sampler.type != V4L2_TUNER_SDR )
		  || ( sampler.rangelow > 2048000 )
		  || ( sampler.rangehigh < 2048000 ) ) {
			continue;
		}

		// On SDR's tuner #1 is the actual RF center frequency.
		memset( &tuner,   0, sizeof( struct v4l2_tuner ) );
		tuner.index = 1;

		if ( ioctl( handle, VIDIOC_G_TUNER, &tuner ) == -1 ) {
			continue;
		}

		// Make sure we can actually cover the FM radio band, our main goal
		if ( ( tuner.type != V4L2_TUNER_RF )
		  || ( ( tuner.capability & V4L2_TUNER_CAP_1HZ ) == 0 )
		  || ( tuner.rangelow  >  88100000 )
		  || ( tuner.rangehigh < 107900000 ) ) {
			continue;
		}

		// At this point we've found a valid choice for SDR, so let's configure it

		// Reduce restart time by only setting the sampling rate if needed
		// as the devices tend to have a multi-second latency to do so.
		memset( &scratch_freq, 0, sizeof( scratch_freq ) );
		scratch_freq.type = V4L2_TUNER_SDR;
		scratch_freq.tuner = 0;

		if ( ioctl( handle, VIDIOC_G_FREQUENCY, &scratch_freq ) != 0 ) {
			continue;
		}

		// The only pre-flight requirement, as mentioned above: 2.048MBit sampling rate
		if ( scratch_freq.frequency != 2048000 ) {
			memset( &scratch_freq, 0, sizeof( scratch_freq ) );
			scratch_freq.type = V4L2_TUNER_SDR;
			scratch_freq.tuner = 0;
			scratch_freq.frequency = 2048000;

			if ( ioctl( handle, VIDIOC_S_FREQUENCY, &scratch_freq ) != 0 ) {
				continue;
			}
		}

		break;
	}

	closedir( devices );

	return handle;
}
