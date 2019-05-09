This is a minimal (but usable) example of the SDR interface the Linux Kernel has, without relying on large external projects or complex libraries.

The FM decoder pass is written in SSE3/AVX2 including a relatively accurate and generalized vectorized ATAN2 function which may also be of interest to others. Generally the USB subsystem uses more CPU than this project does when playing FM radio.

Feel free to re-use any and all of this code for any project you wish. I chose the MIT license for a reason, afterall.

Usage is very simple: The program accepts a single argument, the frequency (in the standard I'm used to here in the US of "102.1" for example) and attempts to autodetect and find the appropriate SDR dongle. It was tested with the RTL-SDR dongle, and sits in the sub-1% CPU usage when running.

While the player is running you can use the standard V4L2 utility v4l2-ctl to change the radio station on the fly; presuming the device is /dev/swradio0 (it usually is unless you have multiple devices) you would use...

```
v4l2-ctl -d /dev/swradio0 --tuner-index=1 -f 102.1
```

...to change the station to 102.1 on the fly. I have no immediate plans to add any user interface to this utility, as the codebase is meant to be kept as simple as possible while still using directly usable.