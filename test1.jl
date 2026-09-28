#=
atom_realtime.jl  -  Real-time GPU electron cloud in Julia (GLMakie / OpenGL)
=============================================================================

The Julia version of the atom simulator. Julia is compiled, so the
rejection sampling runs at C-like speed, and GLMakie renders the point
cloud on the GPU via OpenGL - a real-time interactive window, not a PNG.

Only one dependency (GLMakie). The hydrogen wavefunctions are implemented
directly: the generalized Laguerre polynomial by its explicit sum, and the
angular parts as their real Cartesian forms - so no special-function
package is needed.

Setup (first run compiles GLMakie, which takes a minute):
    julia
    julia> import Pkg; Pkg.add("GLMakie")

Run:
    julia atom_realtime.jl
    julia atom_realtime.jl 3dz2 400000        # start orbital, point count

Controls
--------
    mouse drag ....... orbit          scroll ........... zoom
    LEFT / RIGHT ..... switch orbital  SPACE ............ toggle auto-spin
    +/- .............. more/fewer pts  R ................ resample
    ESC / Q .......... quit
=#

using GLMakie
using Printf

# ---------------------------------------------------------------------
# Physics  (atomic units, Bohr radius a0 = 1, Z = 1)
# ---------------------------------------------------------------------

"Generalized Laguerre polynomial L_n^(alpha)(x) via its explicit sum."
function genlaguerre(n::Int, alpha::Int, x::Float64)
    s = 0.0
    for k in 0:n
        s += (-1.0)^k * binomial(n + alpha, n - k) * x^k / factorial(k)
    end
    return s
end

"Radial part R_nl(r) (overall normalization dropped - it cancels in sampling)."
function radial(n::Int, l::Int, r::Float64)
    rho = 2r / n
    return exp(-r / n) * rho^l * genlaguerre(n - l - 1, 2l + 1, rho)
end

# An orbital: name, quantum numbers n,l, and its real angular shape in
# Cartesian form (unnormalized - constants cancel in rejection sampling).
struct Orbital
    name::String
    n::Int
    l::Int
    ang::Function          # (x,y,z,r) -> Float64
end

const ORBITALS = Orbital[
    Orbital("1s",     1, 0, (x, y, z, r) -> 1.0),
    Orbital("2s",     2, 0, (x, y, z, r) -> 1.0),
    Orbital("2pz",    2, 1, (x, y, z, r) -> z / r),
    Orbital("2px",    2, 1, (x, y, z, r) -> x / r),
    Orbital("3s",     3, 0, (x, y, z, r) -> 1.0),
    Orbital("3pz",    3, 1, (x, y, z, r) -> z / r),
    Orbital("3dz2",   3, 2, (x, y, z, r) -> (2z^2 - x^2 - y^2) / r^2),
    Orbital("3dxy",   3, 2, (x, y, z, r) -> (x * y) / r^2),
    Orbital("3dxz",   3, 2, (x, y, z, r) -> (x * z) / r^2),
    Orbital("3dx2y2", 3, 2, (x, y, z, r) -> (x^2 - y^2) / r^2),
    Orbital("4fz3",   4, 3, (x, y, z, r) -> z * (2z^2 - 3x^2 - 3y^2) / r^3),
]

"Probability density |psi|^2 at a point."
@inline function density(o::Orbital, x::Float64, y::Float64, z::Float64)
    r = sqrt(x * x + y * y + z * z)
    r = r == 0.0 ? 1e-12 : r
    psi = radial(o.n, o.l, r) * o.ang(x, y, z, r)
    return psi * psi
end

# ---------------------------------------------------------------------
# Monte-Carlo rejection sampling  (fast native loop - Julia's strength)
# ---------------------------------------------------------------------
function sample_cloud(o::Orbital, npoints::Int)
    extent = 2.2 * o.n^2 + 6.0

    dmax = 0.0                                  # estimate the peak density
    for _ in 1:60_000
        x = (rand() * 2 - 1) * extent
        y = (rand() * 2 - 1) * extent
        z = (rand() * 2 - 1) * extent
        d = density(o, x, y, z)
        dmax = d > dmax ? d : dmax
    end
    dmax *= 1.05

    pts  = Vector{Point3f}(undef, npoints)
    vals = Vector{Float32}(undef, npoints)
    got = 0
    while got < npoints
        x = (rand() * 2 - 1) * extent
        y = (rand() * 2 - 1) * extent
        z = (rand() * 2 - 1) * extent
        d = density(o, x, y, z)
        if rand() * dmax < d
            got += 1
            pts[got]  = Point3f(x, y, z)
            vals[got] = Float32(d)
        end
    end
    return pts, vals, extent
end

# ---------------------------------------------------------------------
# Real-time viewer
# ---------------------------------------------------------------------
function run(; start::String = "2pz", npoints::Int = 250_000)
    idx = something(findfirst(o -> o.name == start, ORBITALS), 3)
    state = Dict(:idx => idx, :n => npoints)
    spin = Ref(true)

    o = ORBITALS[idx]
    pts, vals, extent = sample_cloud(o, npoints)

    positions = Observable(pts)
    colorvals = Observable(vals ./ maximum(vals))
    title     = Observable(@sprintf("%s  |  %d points", o.name, length(pts)))

    fig = Figure(size = (900, 900), backgroundcolor = :black)
    ax  = Axis3(fig[1, 1]; aspect = :data, backgroundcolor = :black,
                title = title, titlecolor = :white, titlesize = 20)
    hidedecorations!(ax)

    scatter!(ax, positions; color = colorvals, colormap = :inferno,
             markersize = 3, markerspace = :pixel, transparency = true)

    function load()
        o = ORBITALS[mod1(state[:idx], length(ORBITALS))]
        pts, vals, _ = sample_cloud(o, state[:n])
        positions[] = pts
        colorvals[] = vals ./ maximum(vals)
        title[] = @sprintf("%s  |  %d points  |  <-/->  switch   space  spin   +/-  density",
                           o.name, length(pts))
    end

    screen = display(fig)

    on(events(fig).keyboardbutton) do ev
        ev.action == Keyboard.press || return
        k = ev.key
        if k == Keyboard.right
            state[:idx] += 1; load()
        elseif k == Keyboard.left
            state[:idx] -= 1; load()
        elseif k == Keyboard.space
            spin[] = !spin[]
        elseif k == Keyboard.equal || k == Keyboard.kp_add
            state[:n] = min(1_500_000, round(Int, state[:n] * 1.5)); load()
        elseif k == Keyboard.minus || k == Keyboard.kp_subtract
            state[:n] = max(20_000, round(Int, state[:n] / 1.5)); load()
        elseif k == Keyboard.r
            load()
        elseif k == Keyboard.escape || k == Keyboard.q
            GLMakie.closeall()
        end
    end

    @async while isopen(screen)          # auto-spin
        spin[] && (ax.azimuth[] += 0.008)
        sleep(1 / 60)
    end

    wait(screen)                          # keep window open until closed
    return fig
end

# ---------------------------------------------------------------------
# CLI:  julia atom_realtime.jl [orbital] [npoints]
# ---------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    start   = length(ARGS) >= 1 ? ARGS[1] : "2pz"
    npoints = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 250_000
    run(; start = start, npoints = npoints)
end