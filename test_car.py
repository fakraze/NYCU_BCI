import keyboard
import serial
import time
import os

#### Must be change to suitable port [Bluetooth Outgoing]
def main():
    import argparse
    parser = argparse.ArgumentParser(description="CECNL BCI 2023 Car Demo")
    parser.add_argument("port_num", type=str, help="Arduino bluetooth serial port")
    args = parser.parse_args()

    ser = serial.Serial(args.port_num, 9600, timeout=1, write_timeout=1)

    while ser.is_open:
        print("\r{}".format(' ' * 30), end="", flush=True)
        if keyboard.is_pressed('w'):
            print("\rforward", end="", flush=True)
            ser.write(b'1')
        elif keyboard.is_pressed('s'):
            print("\rbackward", end="", flush=True)
            ser.write(b'2')
        elif keyboard.is_pressed('a'):
            print("\rleft", end="", flush=True)
            ser.write(b'3')
        elif keyboard.is_pressed('d'):
            print("\rright", end="", flush=True)
            ser.write(b'4')
        elif keyboard.is_pressed('i'):
            print("\rstraight speedup", end="", flush=True)
            ser.write(b'5')
        elif keyboard.is_pressed('k'):
            print("\rstraight speeddown", end="", flush=True)
            ser.write(b'6')
        elif keyboard.is_pressed('o'):
            print("\rturn speedup", end="", flush=True)
            ser.write(b'7')
        elif keyboard.is_pressed('l'):
            print("\rturn speeddown", end="", flush=True)
            ser.write(b'8')
        else:
            print("\r...", end="", flush=True)
            ser.write(b'0')
        time.sleep(0.2)
    print("Q_Q: connection doko???")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        exit(0)
