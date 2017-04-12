import numpy
import math
import time
from scipy.io import loadmat
from hashlib import sha1

#------------------------------------------------------------------------------            
"""
Set constant settings
- query radius (R)
- constant (C) but not used here
- distance function
- number of data points (N)
- number of dimensions (D)
- HLL error for estimating the output size
- LSH error (prob. of not reporting the right answer)
- CPU Ratio (complexity ratio between removing duplicate and computing distance)
- P1 is computed regarding the radius
- Prime (used in universe hashing)
"""
def getR(): return 0.95
def getC(): return 2
def getType(): return "cosine"
def getN(): return 349950
def getD(): return 254
def getHLLError(): return 0.1
def getLSHError(): return 0.1
def getCPURatio(): return 10
def getP1(): return 1 - math.acos(getR())/math.pi
def getPrime(): return 15485863  
    
#------------------------------------------------------------------------------        
"""
Get Alpha for HLL
"""
def getAlpha(p):
    
    if not (4 <= p <= 16):
        raise ValueError("p=%d should be in range [4 : 16]" % p)

    if p == 4:
        return 0.673

    if p == 5:
        return 0.697

    if p == 6:
        return 0.709

    # for m >= 128 or p >= 7
    return 0.7213 / (1.0 + 1.079 / (1 << p)) #m = 2^p = 1 << p
#------------------------------------------------------------------------------            
"""
Get \rho  for HLL
"""
def getRho(w, max_width):
    
    rho = max_width - w.bit_length() + 1 # max_width = 64 - p    

    if rho <= 0:
        raise ValueError('w overflow')

    return rho
#------------------------------------------------------------------------------                
"""
Set parameters for HLL
- *m* governs the space and time complexity of estimating output size
"""
def getHLLParam():

    """
    error_rate = 1.04 / sqrt(m) => m = (1.04/error_rate)^2 
    m = 2 ** p => p = log2(m)
    M(1)... M(m) = 0
    """
    p = int(math.ceil(math.log((1.04 / getHLLError()) ** 2, 2)))
    alpha = getAlpha(p)           
    m = 1 << p
    
    return (p, alpha, m)
#------------------------------------------------------------------------------            
"""
Create LSH infomation
- Return a list of random vectors in N(0, 1)
"""     
def buildLSHsFunction():
    
    # parameter setting
    L = 50 # number of tables
    K = math.ceil(math.log(1 - 0.1**(1/L)) / math.log(getP1())) #number of hash functions (random projections)    
 
    print("# hash tables: ", L)
    print("# random projections: ", K, "\n")
    
    # LSH structure        
    LSHs = [0] * L
     
    for i in range(L):        
        LSHs[i] = numpy.random.standard_normal(size=(getD(), K))
             
    return LSHs        
    
#------------------------------------------------------------------------------                
"""
Hash computation for the whole data and query
- Simply computing the dot product and convert to binary
- Return hash values in integer for easily building the second hash index
"""
def hashComputation(LSH, data):    
    
    # data = D x 1 (sparse matrix CSC)
    # cauchy = K x D    
    # uniform = 1 x K
    # universal = K x 1
    hashValue = numpy.dot(data, LSH)
    hashValue[hashValue >= 0] = 1
    hashValue[hashValue < 0] = 0
        
    return int(sha1(hashValue.tobytes()).hexdigest()[:16], 16) % 15485863 % 349950  
        
#------------------------------------------------------------------------------                
"""
Build hash tables
- Return TABLEs and HLLs both of size L x N
- There should be some empty cells (buckets) and some large cells (buckets)
"""            
def buildDataStructure(LSHs, data):
    
    N = getN()
    L = len(LSHs)
    
    # HyperLogLog params    
    p, alpha, m = getHLLParam()
	
    TABLEs = numpy.empty( (L, N), dtype=object)    
    HLLs = numpy.empty( (L, N), dtype=object)
    
    # Generate random integer of 32 bits
    #hashHLL = numpy.random.random_integers(-2147483648, 2147483646, N) + 2147483648    
    
    for idxTable in range(L):

        for idxPoint in range(N):
            
            # get the hash value of bucket            
            hashValue = hashComputation(LSHs[idxTable], data[idxPoint])
            
            # If bucket is empty, creat a set() and insert point into set
            if not TABLEs[idxTable][hashValue]:
                
                # bucket is a set
                TABLEs[idxTable][hashValue] = set()
                TABLEs[idxTable][hashValue].add(idxPoint)
                
                # HLL is an array of HLL of size m and count
                HLLs[idxTable][hashValue] = [0]*(m+1)
                
            else:    
                # Insert point into the initialized bucket
                TABLEs[idxTable][hashValue].add(idxPoint) # insert into bucket
                
            # Modify HLLS
            HLLs[idxTable][hashValue][m] = HLLs[idxTable][hashValue][m] + 1 #increase count
                
            # Insert into HLLs
            x = int(sha1(bytes(idxPoint)).hexdigest()[:16], 16) #random hash value to get binary string
            # x = int(hashHLL[idxPoint])
            j = x & (m - 1) # get the last p = log2(m) bits as the index of HLL
            w = x >> p # ignore the last p bits and get the position of the left-most 1s
            rho = getRho(w, 64 - p)
			
            if rho > HLLs[idxTable][hashValue][j]:
                HLLs[idxTable][hashValue][j] = rho # update max rho(w)
            
    return TABLEs, HLLs    
#------------------------------------------------------------------------------        
"""
Estimate output size and compute number of collisions
"""    
def outputSizeEstimate(HLLs, hashVALUEs):
    
    L = numpy.size(hashVALUEs)
    numCollision = 0
    
    # Estimate output size    
    p, alpha, m = getHLLParam()    
    maxHLL = [0] * m
    
    for idxTable in range(L):
	
        hashValue = hashVALUEs[idxTable]
		
        if not HLLs[idxTable][hashValue] is None: # in there are points in the bucket
           
           # Update HLL
           maxHLL = numpy.maximum(maxHLL, HLLs[idxTable][hashValue][0 : m])           
           # Get number of collisions
           numCollision = numCollision + HLLs[idxTable][hashValue][m]
    
    # E = alpha * m^2 / (2^(-M[1]) + 2^(-M[2]) + ... + 2^(-M[m]))
    E = alpha * (m ** 2) / sum(math.pow(2.0, -x) for x in maxHLL)
    
    # Correct HLL estimators    
    if E <= 2.5*m: # small range correction
	
	    # number of registers equal to 0
        v = m - numpy.count_nonzero(maxHLL) 
		
        if v != 0:
            E = m * math.log(m/v) # similar to Bins and balls model
			
    elif E > (1 << 32)/30: # large range correction
        E = -(1 << 32) * math.log(1 - E/(1 << 32))
        
    return E, numCollision
#------------------------------------------------------------------------------        
"""
r-near neighbor query with Output-sensitive LSH
"""    
def rNN_OutLSH(TABLEs, HLLs, LSHs, query, data):
    
    L = len(LSHs)    
    hashVALUEs = [0] * L
        
    #start = time.clock()    
    for idxTable in range(L):	
        
        hashVALUEs[idxTable] = hashComputation(LSHs[idxTable], query)

    #print("Hash Computation time: ", time.clock() - start, " in second")
        
    # Estimate output size        
    #start = time.clock()
    candidateSizeEst, numCollision = outputSizeEstimate(HLLs, hashVALUEs)
    
    #print("Output Size Estimation Time: ", time.clock() - start, " in second")
    #print("Candidate Size Estimate:", candidateSizeEst, " Number of Collisions: ", numCollision)
    
    # Searching    
    rNN = list()
    N = getN()    
    useLinear = False
    
    # We need to make sure the ratio getCPURatio() correctly. 
    # It refers to the complexity ratio between removing duplicate and computing distance
    if candidateSizeEst + numCollision / getCPURatio() > N : 

        useLinear = True        
        #print("Use Linear Search for OutputSensitive LSH...")
        for idxPoint in range(N):        
            
           temp = query - data[idxPoint]
           if numpy.dot(temp, temp.T) >= 0.95: # use 0.95 is faster than calling getR()
               rNN.append(idxPoint)
        
    else:        
        
        #print("Use standard LSH for OutputSensitive LSH...")  
        candidateNN = set();

        # Remove duplicates        
        for idxTable in range(L):
            
            hashValue = hashVALUEs[idxTable]
        
            if TABLEs[idxTable][hashValue]:
                
                for idxPoint in TABLEs[idxTable][hashValue]:
                    candidateNN.add(idxPoint)
                # candidateNN = candidateNN.union(TABLEs[idxTable][hashValue])
        
        # Compute distance
        for idxPoint in candidateNN:
		
            temp = query - data[idxPoint]
            if numpy.dot(temp, temp.T) >= 0.95:  # use 0.95 is faster than calling getR()
               rNN.append(idxPoint)
        
    return rNN, useLinear#, candidateSizeEst, numCollision
	
#------------------------------------------------------------------------------        
"""
r-near neighbor query with LSH
"""    
def rNN_LSH(TABLEs, LSHs, query, data):    
    
    L = len(LSHs)    
    candidateNN = set()     
        
    # Hash computation
    hashVALUEs = [0] * L    
    
    for idxTable in range(L):	        
        hashVALUEs[idxTable] = hashComputation(LSHs[idxTable], query)
        
	# Removing duplicates    
    #start = time.clock()
    for idxTable in range(L):
        
        hashValue = hashVALUEs[idxTable]
        
        if TABLEs[idxTable][hashValue]:
            
            #candidateNN = candidateNN.union(TABLEs[idxTable][hashValue])
            
            for idxPoint in TABLEs[idxTable][hashValue]:
                candidateNN.add(idxPoint)
			
    #duplicateTime = time.clock() - start
    
    # Compute distance
    rNN = list()
    for idxPoint in candidateNN:
        
        temp = query - data[idxPoint]
        if numpy.dot(temp, temp.T) >= 0.95:
           rNN.append(idxPoint)
        
    return rNN#, duplicateTime, len(candidateNN)
#------------------------------------------------------------------------------        
"""
r-near neighbor query with Linear Search
"""    
def rNN_Linear(query, data):
    
    # Searching    
    rNN = list()    
        
    for idxPoint in range(getN()): 
	
        temp = query - data[idxPoint]
        if numpy.dot(temp, temp.T) >= 0.95:      
           rNN.append(idxPoint)    
           
    return rNN    

#------------------------------------------------------------------------------                    
"""
Main function for Webspam dataset with 50 queries
"""
data = loadmat("/mnt/fast_storage/users/ndap/Dataset/_LSH/Real/webspam/webspam_data.mat")
query = loadmat("/mnt/fast_storage/users/ndap/Dataset/_LSH/Real/webspam/webspam_query.mat")

#data = loadmat("C:\_Data\Dataset\_LSH\Real\webspam\webspam_data.mat")
#query = loadmat("C:\_Data\Dataset\_LSH\Real\webspam\webspam_query.mat")

query = query['query'].transpose().toarray()
data = data['data'].transpose().toarray()
#data = data[0 : 50000]

numQuery = 50

#-----------------------------------------------------------

LSHs = buildLSHsFunction()
TABLEs, HLLs = buildDataStructure(LSHs, data)

#-----------------------------------------------------------

start = time.clock()
for i in range(numQuery):

    rNN = rNN_LSH(TABLEs, LSHs, query[i], data)
	
    
print("LSH Time: ", time.clock() - start, " in second")

#-----------------------------------------------------------

LS = 0
start = time.clock()
for i in range(numQuery):

     rNN, useLinear = rNN_OutLSH(TABLEs, HLLs, LSHs, query[i], data)
	 
         
     if useLinear: LS = LS + 1
    
print("Output-sensitive LSH Time: ", time.clock() - start, " in second")
print("Number of Linear Search: ", LS)


#-----------------------------------------------------------

start = time.clock()
for i in range(numQuery):

    rNN = rNN_Linear(query[i], data)

   
print("Linear Time: ", time.clock() - start, " in second \n")

