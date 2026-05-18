command to compile the converter:
`gcc -o naf naf_converter.c -lm`

there are three modes (only used in encode):
-Q (Quality)
-S (Size)
-Q9 (Maximum Quality)
all commands: 
./naf encode -Q/-S/-Q9 oldfile.wav newfile.naf
./naf status newfile.naf
./naf decode newfile.naf newerfile.wav

