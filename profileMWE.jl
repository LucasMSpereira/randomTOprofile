using TimerOutputs, Ferrite, TopOpt, Parameters, Dates
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent
include("./utils.jl")
import Nonconvex
Nonconvex.@load NLopt
const to = TimerOutput()
reset_timer!(to)

function methodThroughput(nSample::Int)
  for sample in 1:nSample
    println("Sample $sample   $(timeNow())")
    problem, vf, _ = randomFEAproblem(FEAparams) # random problem
    solver = FEASolver(Direct, problem; xmin = 1e-6, penalty = TopOpt.PowerPenalty(3.0))
    solver()
    comp = TopOpt.Compliance(problem, solver) # compliance
    filter = DensityFilter(solver; rmin = 3.0) # filtering to avoid checkerboard
    obj = x -> comp(filter(x)) # objective
    x0 = fill(vf, FEAparams.nElements) # starting densities (VF everywhere)
    volfrac = TopOpt.Volume(problem, solver)
    constr = x -> volfrac(filter(x)) - vf # volume fraction constraint
    model = Nonconvex.Model(obj) # create optimization model
    Nonconvex.addvar!( # add optimization variable
      model, zeros(FEAparams.nElements), ones(FEAparams.nElements), init = x0
    )
    Nonconvex.add_ineq_constraint!(model, constr) # add volume constraint
    @timeit to "standard" Nonconvex.optimize(model, NLoptAlg(:LD_MMA), x0; options = NLoptOptions())
  end
end

methodThroughput(100)

show(to)