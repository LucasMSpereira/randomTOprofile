using TimerOutputs, Ferrite, TopOpt, Parameters
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent
import Nonconvex
Nonconvex.@load NLopt
const to = TimerOutput()
reset_timer!(to)

function randDiffInt(n, val)
  randVec = zeros(Int, n)
  randVec[1] = rand(1:val)
  for ind in 2:n
    randVec[ind] = rand(1:val)
    while in(randVec[ind], randVec[1:ind-1])
      randVec[ind] = rand(1:val)
    end
  end
  return randVec
end

function quad(nelx::Int, nely::Int, vec::Vector{<:Real})
  quad_ = zeros(nely, nelx)
  for i in 1:nely
    for j in 1:nelx
      quad_[nely - (i - 1), j] = vec[(i - 1) * nelx + 1 + (j - 1)]
    end
  end
  return quad_
end

# struct with parameters
Parameters.@with_kw mutable struct FEAparameters
  quants::Int = 1 # number of TO problems per section
  V::Array{Real} = [0.4 + rand() * 0.5 for i in 1 : quants] # volume fractions
  problems::Any = Array{Any}(undef, quants) # store FEA problem structs
  meshSize::Tuple{Int, Int} = (140, 50) # Size of rectangular mesh
  section::Int = 1 # Number of dataset HDF5 files with "quants" samples each
  nElements::Int32 = prod(meshSize) # quantity of elements
  nNodes::Int32 = prod(meshSize .+ 1) # quantity of nodes
  # matrix with element IDs in their respective position in the mesh
  elementIDmatrix::Array{Int, 2} = convert.(Int, quad(meshSize..., [i for i in 1:nElements]))
  elementIDarray::Array{Int} = [i for i in 1:nElements] # Vector that lists element IDs
  meshMatrixSize::Tuple{Int, Int} = (51, 141) # Size of rectangular nodal mesh as a matrix
end
FEAparams = FEAparameters()

# Create vector of (float, float) tuples with node coordinates for "node_coords"
# Supposes rectangular elements with unit sides staring at postition (0.0, 0.0)
# meshSize = (x, y) = quantity of elements in each direction
function mshData(meshSize)
  coordinates = Array{Tuple{Float64, Float64}}(undef, (meshSize[1]+1)*(meshSize[2]+1))
  for line in 1:(meshSize[2] + 1)
    coordinates[(line + (line - 1) * meshSize[1]) : (line * (1 + meshSize[1]))] .= [((col - 1)/1, (line - 1)/1) for col in 1:(meshSize[1] + 1)]
  end
  g_num = Array{Tuple{Vararg{Int, 4}}, 1}(undef, prod(meshSize))
  for elem in 1:prod(meshSize)
    dd = floor(Int32, (elem - 1)/meshSize[1]) + elem
    g_num[elem] = (dd, dd + 1, dd + meshSize[1] + 2, dd + meshSize[1] + 1)
  end
  return coordinates, g_num
end

# Create the node set necessary for specific and well defined support conditions
function simplePins!(type, dispBC, FEAparams)
  type == "rand" && (type = rand(["left" "right" "top" "bottom"]))
  if type == "left" # Clamp left boundary of rectangular domain.
    fill!(dispBC, 4)
    firstCol = [(n-1)*(FEAparams.meshSize[1]+1) + 1 for n in 1:(FEAparams.meshSize[2]+1)]
    secondCol = firstCol .+ 1
    nodeSet = Dict("supps" => vcat(firstCol, secondCol))
  elseif type == "right" # Clamp right boundary of rectangular domain.
    fill!(dispBC, 6)
    firstCol = [(FEAparams.meshSize[1]+1)*n for n in 1:(FEAparams.meshSize[2]+1)]
    secondCol = firstCol .- 1
    nodeSet = Dict("supps" => vcat(firstCol, secondCol))
  elseif type == "bottom" # Clamp bottom boundary of rectangular domain
    fill!(dispBC, 5)
    nodeSet = Dict("supps" => [n for n in 1:(FEAparams.meshSize[1]+1)*2])
  elseif type == "top" # Clamp top boundary of rectangular domain
    fill!(dispBC, 7)
    nodeSet = Dict("supps" => [n for n in ((FEAparams.meshSize[1]+1)*(FEAparams.meshSize[2]-1)+1):((FEAparams.meshSize[1]+1)*((FEAparams.meshSize[2]+1)))])
  end
  return nodeSet, dispBC
end

# create random FEA problem
function randomFEAproblem(FEAparams)
  elType = "CPS4"
  if elType == "CPS4"
    grid = generate_grid(Quadrilateral, FEAparams.meshSize)
  elseif elType == "CPS8"
    grid = generate_grid(QuadraticQuadrilateral, FEAparams.meshSize)
  end
  numCellNodes = length(grid.cells[1].nodes) # number of nodes per cell/element
  nels = prod(FEAparams.meshSize) # number of elements in the mesh
  # nodeCoords = Vector of tuples with node coordinates
  # cells = Vector of tuples of integers. Each line refers to an element
  # and lists the IDs of its nodes
  nodeCoords, cells = mshData(FEAparams.meshSize)
  # Similar to nodeSets, but refers to groups of cells (FEA elements) 
  cellSets = Dict("SolidMaterialSolid" => FEAparams.elementIDarray,"Eall" => FEAparams.elementIDarray,"Evolumes"=> FEAparams.elementIDarray)
  dispBC = zeros(Int, (3,3))
  # nodeSets = dictionary mapping strings to vectors of integers. The vector groups 
  # node IDs that can be later referenced by the name in the string
  if rand() > 0.6 # "clamp" a side
      nodeSets, dispBC = simplePins!("rand", dispBC, FEAparams)
  else # position pins randomly
      nodeSets, dispBC = randPins!(nels, FEAparams, dispBC, grid)
  end
  lpos, forces = loadPos(nels, dispBC, FEAparams, grid)
  cLoads = Dict(lpos[1] => forces[1,3:4])
  [merge!(cLoads, Dict(lpos[c] => forces[1,3:4])) for c in 2:numCellNodes];
  if length(lpos) > numCellNodes+1
      for pos in (numCellNodes+1):length(lpos)
          pos == (numCellNodes+1) && (global ll = 2)
          merge!(cLoads, Dict(lpos[pos] => forces[ll,3:4]))
          pos % numCellNodes == 0 && (global ll += 1)
      end
  end
  vf = randBetween(0.3, 0.9)[1]
  return InpStiffness(
      InpContent(
          nodeCoords, elType, cells, nodeSets, cellSets,  vf * 210e3, 0.3,
          0.0, Dict("supps" => [(1, 0.0), (2, 0.0)]), cLoads,
          Dict("uselessFaces" => [(1,1)]), Dict("uselessFaces" => 0.0)
      )
  ), vf, forces
end

function loadPos(nels, dispBC, FEAparams, grid)
  # Random ID(s) to choose element(s) to be loaded
  loadElements = randDiffInt(2, nels)
  # Matrices to indicate position and component of load
  forces = zeros(2,4)'
  # i,j mesh positions of chosen elements
  loadPoss = findall(x -> in(x, loadElements), FEAparams.elementIDmatrix)
  
  # Verify if load will be applied on top of support.
  # Randomize positions again if that's the case
  while true
    if dispBC[1,3] > 3


      if dispBC[1,3] == 4
        # left
        if prod([loadPoss[i][2] != 1 for i in keys(loadPoss)])
          break
        else
          loadElements = randDiffInt(2, nels)
          loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
        end
      elseif dispBC[1,3] == 5
        # bottom
        if prod([loadPoss[i][1] != FEAparams.meshSize[2] for i in keys(loadPoss)])
          break
        else
          loadElements = randDiffInt(2, nels)
          loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
        end
      elseif dispBC[1,3] == 6
        # right
        if prod([loadPoss[i][2] != FEAparams.meshSize[1] for i in keys(loadPoss)])
          break
        else
          loadElements = randDiffInt(2, nels)
          loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
        end
      elseif dispBC[1,3] == 7
        # top
        if prod([loadPoss[i][1] != 1 for i in keys(loadPoss)])
          break
        else
          loadElements = randDiffInt(2, nels)
          loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
        end
      end


    else


      boolPos = true
      for i in keys(loadPoss)
        boolPos *= !in([loadPoss[i][k] for k in 1:2], [dispBC[h,1:2] for h in 1:size(dispBC)[1]])
      end
      if boolPos
        break
      else
        loadElements = randDiffInt(2, nels)
        loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
      end


    end
  end
  # Generate point load component values
  randLoads = (-ones(length(loadElements),2) + 2*rand(length(loadElements),2))*90
  # Build matrix with positions and components of forces
  forces = [
    loadPoss[1][1] loadPoss[1][2] randLoads[1,1] randLoads[1,2]
    loadPoss[2][1] loadPoss[2][2] randLoads[2,1] randLoads[2,2]
  ]
  # Get vector with IDs of loaded nodes
  myCells = [grid.cells[g].nodes for g in loadElements]
  pos = reshape([myCells[ele][eleNode] for eleNode in 1:length(myCells[1]), ele in keys(loadElements)], (:,1))
  return pos, forces, randLoads
end

function randPins!(nels, FEAparams, dispBC, grid)
  # generate random element IDs
  randEl = randDiffInt(3, nels)
  # get "matrix position (i,j)" of elements chosen
  suppPos = findall(x->in(x,randEl), FEAparams.elementIDmatrix)
  # build compact dispBC with pin positions chosen
  for pin in 1:length(unique(randEl))
    dispBC[pin,1] = suppPos[pin][1]
    dispBC[pin,2] = suppPos[pin][2]
    dispBC[pin,3] = 3
  end
  # get node positions of pins
  myCells = [grid.cells[g].nodes for g in randEl]
  pos = vec(reshape([myCells[ele][eleNode] for eleNode in 1:length(myCells[1]), ele in keys(randEl)], (:,1)))
  nodeSets = Dict("supps" => pos)
  return nodeSets, dispBC
end

function methodThroughput(nSample::Int)
  for sample in 1:nSample
    # sample % round(Int, nSample/5) == 0 && @show sample
    @show sample
    while true
      print(1, "  ")
      problem, vf, _ = randomFEAproblem(FEAparams) # random problem
      print(2, "  ")
      solver = FEASolver(Direct, problem; xmin = 1e-6, penalty = TopOpt.PowerPenalty(3.0))
      print(3, "  ")
      solver()
      print(4, "  ")
      comp = TopOpt.Compliance(solver) # compliance
      print(5, "  ")
      filter = DensityFilter(solver; rmin = 3.0) # filtering to avoid checkerboard
      print(6, "  ")
      obj = x -> comp(filter(PseudoDensities(x))) # objective
      print(7, "  ")
      x0 = fill(vf, FEAparams.nElements) # starting densities (VF everywhere)
      print(8, "  ")
      volfrac = TopOpt.Volume(solver)
      print(9, "  ")
      constr = x -> volfrac(filter(PseudoDensities(x))) - vf # volume fraction constraint
      print(10, "  ")
      model = Nonconvex.Model(obj) # create optimization model
      print(11, "  ")
      Nonconvex.addvar!( # add optimization variable
        model, zeros(FEAparams.nElements), ones(FEAparams.nElements), init = x0
      )
      Nonconvex.add_ineq_constraint!(model, constr) # add volume constraint
      @timeit to "standard" Nonconvex.optimize(model, NLoptAlg(:LD_MMA), x0; options = NLoptOptions())
    end
  end
end

methodThroughput(150)

show(to)