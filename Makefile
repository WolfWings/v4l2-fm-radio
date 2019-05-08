CFLAGS = -O3 -march=native -s -Wall -pedantic -Wextra -ffunction-sections
LDFLAGS = -s -Wl,--gc-sections -Wl,--print-gc-sections

player: player.o sdr.o fm_decoder.o

player.o: player.c sdr.h fm_decoder.h

clean:
	-rm -f player *.o
