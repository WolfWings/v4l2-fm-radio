The kernel requirements to enable the RTL-SDR driver have a couple of "gotcha" items, so I'm documenting them here for completeness.

<pre>
Device Drivers --->
  🗹 I2C support --->
    🗹 I2C bus multiplexing support
  🗹 Multimedia Support --->
    🗹 Digital TV support
    🗹 Software defined radio support
    🗹 Media USB Adapters --->
      🗹 Support for various USB DVB devices v2
        🗹 Realtek RTL28xxU DVB USB support
</pre>

If you've disabled the "Autoselect" option you'll need the following items enabled manually:

<pre>
Customize TV tuners --->
  🗹 Rafael Micro R820T silicon tuner
Customize DVB Frontends --->
  🗹 Realtek RTL2832 SDR
</pre>

The only real "gotcha" items are that the SDR support is tied to enabling the "RTL28xxU" support, which enables the RTL2830 driver (and requires the DVB-T components) as well, and needing to make sure the I2C bus multiplexing is enabled as this chipset relies heavily on it. Just enabling the SDR and R820T are insufficient as they will not be "tied together" without the RTL28xxU driver being present as well.

KERNEL BUILD BUG:

On some kernel builds module support being enabled at all (not just these drivers being built as a module) breaks this driver stack from working at all due to a flaw in a cross-module function call macro. In effect the function is in the kernel, but the macro being used doesn't see that the function exists even if the other module is loaded, so it aborts initializing the driver.

Messages about "r820t_attach" failures in your dmesg are indicative of this bug.
