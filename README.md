# mrhappy

print out [Happy Numbers](https://en.wikipedia.org/wiki/Happy_number) up to 1,000,000 really fast

## build and run

it's CMake remember to set `-DCMAKE_BUILD_TYPE=Release`

only tested on Ryzen 5 5600X

```
cmake -DCMAKE_BUILD_TYPE=Release -B build
cmake --build build
./build/mrhappy
```

## todo

it uses an inefficient `write` loop, but `fputs` is slower

need to make a buffer that fits in L1D to reduce stalled cycles but don't feel like handling that logic right now
