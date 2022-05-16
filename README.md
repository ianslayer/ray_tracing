# What is this

This is a toy ray tracer I wrote for fun. The scene is the Cornell box and is constructed in code.

It has 3 options:

    --sample_count: sample count per pixel
    --bounce_count: max bounce depth of each path
    --method: scalar or simd

# How to build

The only cpp file is ray_tracing/main.cpp. That should be trivial to build in any sufficiently modern C++ compilers.

I also include the project files for Windows and Mac:

For Windows, use Visual Studio newer than 2019, open ray_tracing_msvc/ray_tracing_msvc.sln and build.

For Mac, use Xcode, open ray_tracing.xcodeproj, and build.