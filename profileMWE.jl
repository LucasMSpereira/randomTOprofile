using Revise, TimerOutputs, Ferrite, TopOpt, Parameters, Dates, Zygote
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent
include("./utils.jl")
import Nonconvex
Nonconvex.@load NLopt
const to = TimerOutput()

function construct()
  problem, vf, _ = randomFEAproblem(FEAparams) # random problem
  solver = FEASolver(Direct, problem; xmin = 1e-6, penalty = TopOpt.PowerPenalty(3.0))
  @timeit to "solver" solver()
  comp = TopOpt.Compliance(solver) # compliance
  return comp, solver, problem, vf
end

function methodThroughput(nSample::Int)
  reset_timer!(to)
  for sample in 1:nSample
    sample % 10 == 0 && println("Sample $sample   $(timeNow())")
    comp, solver, _, vf = construct()
    filter = DensityFilter(solver; rmin = 3.0) # filtering to avoid checkerboard
    obj = x -> comp(filter(PseudoDensities(x))) # objective
    x0 = fill(vf, FEAparams.nElements) # starting densities (VF everywhere)
    volfrac = TopOpt.Volume(solver)
    constr = x -> volfrac(filter(PseudoDensities(x))) - vf # volume fraction constraint
    model = Nonconvex.Model(obj) # create optimization model
    Nonconvex.addvar!( # add optimization variable
      model, zeros(FEAparams.nElements), ones(FEAparams.nElements), init = x0
    )
    Nonconvex.add_ineq_constraint!(model, constr) # add volume constraint
    @timeit to "standard" Nonconvex.optimize(model, NLoptAlg(:LD_MMA), x0; options = NLoptOptions())
  end
end

methodThroughput(150)

show(to)