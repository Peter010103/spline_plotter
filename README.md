# Spline Plotter
An OpenGL program for generating and visualising spline curves.

## Preview
<p align="center">
  <img src="figs/screenshot.png" width="800">
</p>

## Execution
After compiling the program with `make`, start the program with
```
./spline_plotter --spline_type {$spline_type} --show_convex_hull {$style} --CCont {} --GCont {}
```

Replace
- `{$spline_type}` with one of the following options `{Hermite, Bezier, BSpline, CatmullRom, MINVO}`
- `{$style}` with `{0, 1, 2}`
  - Selecting `1` will draw lines joining adjacent control points, `2` will draw the control polygons

Note the only required argument is `--spline_type`. The rest are optional.

Whilst running the program, click on the screen to insert control points.

Pressing,
- `<F1>` will clear all previously inserted points
- `<r>` will remove the last inserted point
- `<e>` will export the spline data in the `results` directory

