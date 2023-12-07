timeNow() = replace(string(ceil(now(), Dates.Second)), ":" => "-")[6:end]

# Create hdf5 file. Store data in a more efficient way
function createFile(quants, sec, runID, nelx, nely)
  # create file
  quickTOdata = h5open(datasetPath * "data/$runID $sec $quants", "w")
  # organize data into folders/groups
  create_group(quickTOdata, "inputs")
  # initialize data in groups
  create_dataset(quickTOdata, "topologies", zeros(nely, nelx, quants))
  # volume fraction
  create_dataset(quickTOdata["inputs"], "VF", zeros(quants))
  # representation of mechanical supports
  create_dataset(quickTOdata["inputs"], "dispBoundConds", zeros(Int, (3,3,quants)))
  # location and value of forces
  create_dataset(quickTOdata["inputs"], "forces", zeros(2,4,quants))
  # norm of displacement vector interpolated in the center of each element
  create_dataset(quickTOdata, "disp", zeros(nely, nelx, 2*quants))
  # return file id to write info during dataset generation
  return quickTOdata
end

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
  quants::Int = 50 # number of TO problems per section
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

# calculate stresses, principal components and strain energy density
function calcConds(nels, disp, problemID, e, v, numCellNode)
  # "Programming the finite element method", 5. ed, Wiley, pg 35
  state = "stress"
  # principal stresses
  principals = permutedims(zeros(FEAparams.meshSize..., 2), (2, 1, 3))
  σ = Array{Real}(undef, nels) # stresses
  strainEnergy = zeros(FEAparams.meshSize)' # strain energy density in each element
  vm = zeros(FEAparams.meshSize)' # von Mises for each element
  centerDispGrad = Array{Real}(undef, nels, 2)
  cellValue = CellVectorValues(
    QuadratureRule{2, RefCube}(2), Lagrange{2,RefCube,ceil(Int, numCellNode/7)}()
  )
  el = 1
  # determine stress-strain relationship dee according to 2D stress type
  dee = deeMat(state, e, v)
  # vecDisp = dispVec(disp) # rearrange disp into vector
  # loop in elements
  for cell in CellIterator(FEAparams.problems[problemID].ch.dh)
    reinit!(cellValue, cell)
    # interpolate gradient of displacements on the center of the element
    # centerDispGrad = function_symmetric_gradient(cellValue, 1, vecDisp[celldofs(cell)])
    centerDispGrad = function_symmetric_gradient(cellValue, 1, disp[celldofs(cell)])
    # use gradient components to build strain vector ([εₓ ε_y γ_xy])
    ε = [
      centerDispGrad[1,1]
      centerDispGrad[2,2]
      centerDispGrad[1,2]+centerDispGrad[2,1]
    ]
    # use constitutive model to calculate stresses in the center of current element
    stress = dee*ε
    # take norm of stress vector to associate a scalar to each element
    σ[el] = norm(stress)
    # element cartesian position in mesh
    elPos = findfirst(x->x==el,FEAparams.elementIDmatrix)
    # extract principal stresses
    principals[elPos, :] .= sort(eigvals([stress[1] stress[3]; stress[3] stress[2]]))
    # build matrix with (center) von Mises value for each element
    vm[elPos] = sqrt(stress'*[1 -0.5 0; -0.5 1 0; 0 0 3]*stress)
    strainEnergy[elPos] = (1+v)*(stress[1]^2+stress[2]^2+2*stress[3]^2)/(2*e) - v*(stress[1]+stress[2])^2/(2*e)
    el += 1
  end
  return vm, σ, principals, strainEnergy
end

# determine stress-strain relationship dee according to 2D stress type
function deeMat(state, e, v)
  if state == "strain"
    # plane strain
    dee = e*(1 - v)/((1 + v)*(1 - 2 * v))*
      [1 v/(1 - v) 0;
      v/(1 - v) 1 0;
      0 0 (1 - 2*v)/(2*(1 - v))]
  
  elseif state == "stress"
    # plane stress
    dee = e/(1-v^2)*[
    1 v 0
    v 1 0
    0 0 (1-v)/2
    ]
  elseif state == "axisymmetric"
    
    dee = e*(1 - v)/((1 + v)*(1 - 2 * v))*
    [1 v/(1 - v) 0 v/(1 - v);
    v/(1 - v) 1 0 v/(1 - v);
    0 0 (1 - 2*v)/(2*(1 - v)) 0;
    v/(1 - v) v/(1 - v) 0 1]
  else
    println("Invalid stress state.")
  end
  return dee
end

# check if sample was generated correctly (solved the FEA problem it was given and didn't swap loads)
function checkSample(numForces, vals, quants, forces)
  sProds = zeros(numForces)
  grads = zeros(2,numForces)
  avgs = similar(sProds)
  # get physical quantity gradients and average value around locations of loads
  for f in 1:numForces
    grads[1,f], grads[2,f], avgs[f] = estimateGrads(vals, quants, round.(Int,forces[f,1:2])...)
  end
  # calculate dot product between normalized gradient and respective normalized load to check for alignment between the two
  [sProds[f] = dot((grads[:,f]/norm(grads[:,f])),(forces[f,3:4]/norm(forces[f,3:4]))) for f in 1:numForces]
  # ratio of averages of neighborhood values of scalar field
  vmRatio = avgs[1]/avgs[2]
  # ratio of load norms
  loadRatio = norm(forces[1,3:4])/norm(forces[2,3:4])
  # ratio of the two ratios above
  ratioRatio = vmRatio/loadRatio
  magnitude = false
  # test alignment
  alignment = sum(abs.(sProds)) > 1.8
  if alignment
    # test scalar neighborhood averages against force norms
    magnitude = (ratioRatio < 1.5) && (ratioRatio > 0.55)
  end
  return alignment*magnitude
end

# estimate scalar gradient around element in mesh
function estimateGrads(vals, quants, iCenter, jCenter)
  peaks = Array{Any}(undef,quants)
  Δx = zeros(quants)
  Δy = zeros(quants)
  avgs = 0.0
  # pad original matrix with zeros along its boundaries to avoid index problems with kernel
  cols = size(vals,2)
  lines = size(vals,1)
  vals = vcat(vals, zeros(quants,cols))
  vals = vcat(zeros(quants,cols), vals)
  vals = hcat(zeros(lines+2*quants,quants), vals)
  vals = hcat(vals,zeros(lines+2*quants,quants))
  for circle in 1:quants
    # size of internal matrix
    side = 2*(circle+1) - 1
    # variation in indices
    delta = convert(Int,(side-1)/2)
    # build internal matrix
    mat = vals[(iCenter-delta+quants):(iCenter+delta+quants),(jCenter-delta+quants):(jCenter+delta+quants)]
    # calculate average neighborhood values
    circle == quants && (avgs = mean(filter(!iszero,mat)))
    # nullify previous internal matrix/center element
    if size(mat, 1) < 4
      mat[2, 2] = 0
    else
      mat[2:(end - 1) , 2:(end - 1)] .= 0
    end
    # store maximum value of current ring (and its position relative to the center element)
    peaks[circle] = findmax(mat)
    center = round(Int, (side + 0.01) / 2)
    Δx[circle] = peaks[circle][2][2] - center
    Δy[circle] = center - peaks[circle][2][1]
  end
  maxVals = [peaks[f][1] for f in keys(peaks)]
  x̄ = Δx' * maxVals / sum(maxVals)
  ȳ = Δy' * maxVals / sum(maxVals)
  return x̄, ȳ, avgs

end

# write displacements to file
function writeDispComps(quickTOdata, problemID, disp, FEAparams, numCellNode)
  dispInterp = Array{Real}(undef, prod(FEAparams.meshSize), 2)
  cellValue = CellVectorValues(QuadratureRule{2, RefCube}(2), Lagrange{2, RefCube, ceil(Int, numCellNode/7)}())
  el = 1
  for cell in CellIterator(FEAparams.problems[problemID].ch.dh) # loop in elements
    reinit!(cellValue, cell)
    # interpolate displacement (u, v) of element center based on nodal displacements.
    dispInterp[el, :] = function_value(cellValue, 1, disp[celldofs(cell)])
    el += 1
  end
  # add to dataset
  quickTOdata["disp"][:, :, 2 * problemID - 1] = quad(FEAparams.meshSize..., dispInterp[:, 1])
  quickTOdata["disp"][:, :, 2 * problemID] = quad(FEAparams.meshSize..., dispInterp[:, 2])
  return dispInterp
end