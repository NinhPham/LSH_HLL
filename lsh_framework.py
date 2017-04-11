import numpy
import math
from hashlib import sha1
import time
import pandas
from bitarray import bitarray, bitdiff
from scipy.linalg import hadamard
import scipy.spatial
#------------------------------------------------------------------------------            
"""
Set constant settings
"""
def getR(): return 5
def getC(): return 2
def getType(): return "bit_sampling"
def getN(): return 349950
def getD(): return 64
def getPrime(): return 9973 # int((1 << 31) - 1) # Mersenne prime
def getHLLError(): return 0.2
def getLSHError(): return 0.01
def getCPURatio(): return 1
def getW(): return 16
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
Distance computation
"""     
def dist(x, y):
    
    if getType() == "bit_sampling":
        return bitdiff(x, y)
        
    elif getType() ==  "basic_covering":
        return bitdiff(x, y)
        
    elif getType() == "l1":
        return scipy.spatial.distance.cityblock(x, y)
        
    elif getType() == "l2":
        return scipy.spatial.distance.euclidean(x, y)
        
    elif getType() == "cosine":
        return scipy.spatial.distance.cosine(x, y)
    
#------------------------------------------------------------------------------            
"""
Create LSH infomation
"""     
def buildLSHsFunction():
    
    LSHs = []
    
    if getType() == "bit_sampling":
        
        # parameter setting
        L = 100 #2**(R+1) - 1
        K = math.ceil(math.log(1 - (1 - getLSHError())**(1/L)) / math.log(1 - getR() / getD()))        

        # LSH structure        
        LSHs = [0]*L
		
        for i in range(L):
            LSHs[i] = numpy.random.random_integers(0, getD() - 1, K).tolist()
    
    elif getType() == "basic_covering":
        
        # parameter setting
        L = 2**(getR() + 1) - 1
        LSHs = [getD() * bitarray('0') for _ in range(L)]
        
        # mapping m: {0, ..., D-1} ->{1, ..., L} for removing the trivial 1st position
        mapping = numpy.random.random_integers(1, L, getD()).tolist()
        
        # ignore the 1st row for trivial collision
        HADAMARD = (1 - hadamard(L + 1)) // 2;
        HADAMARD = HADAMARD[1 : , :] 
        
        # randomly sample D collumns
        sampleHAD = HADAMARD[:, mapping] 
        
        # Generate info for each hash table
        for i in range(L):
            LSHs[i] = bitarray(sampleHAD[i, :].tolist())
            
    elif getType() == "l1":
         
        # parameter setting
        L = 100 # getN() ** (1 / getC())
        K = math.ceil(math.log(1 - (1 - getLSHError())**(1/L)) / math.log(1 - getR() / getD()))        
 
        # LSH structure        
        LSHs = [{} for _ in range(L)]
         
        # LSH info
        for i in range(L):
            
            cauchy = numpy.random.standard_cauchy(size=(getD(), K)) 
            uniform = numpy.random.uniform(0, getW(), K)
            universal = numpy.random.random_integers(0, getPrime(), K)
            LSHs[i] = {'cauchy' : cauchy, 'uniform' : uniform, 'universal' : universal}
             
    elif getType() == "l2":
         
        # parameter setting
        L = 100
        K = math.ceil(math.log(1 - (1 - getLSHError())**(1/L)) / math.log(1 - getR() / getD()))
 
        # LSH structure        
        LSHs = [{} for _ in range(L)]
         
        for i in range(L):
            
            gauss = numpy.random.standard_normal(size=(getD(), K)) 
            uniform = numpy.random.uniform(0, getW(), K)           
            universal = numpy.random.random_integers(0, getPrime(), K)
            LSHs[i] = {'gauss' : gauss, 'uniform' : uniform, 'universal' : universal}
             
    elif type == "cosine":
         
        # parameter setting
        L = 100
        K = math.ceil(math.log(1 - (1 - getLSHError())**(1/L)) / math.log(1 - getR() / getD()))
 
        # LSH structure        
        LSHs = [0] * L
         
        for i in range(L):
            LSHs[i] = numpy.random.standard_normal(size=(getD(), K))             
        
    return LSHs
        
    
#------------------------------------------------------------------------------                
"""
Hash computation for the whole data and query
"""
def hashComputation(LSH, data):    
    
    if getType() == "bit_sampling":                   
        
        K = len(LSH)
        hashValue = K * bitarray('0')
        
        for i in range(K):
            
            if data[LSH[i]]:
                hashValue[i] = True
        
        return int(sha1(hashValue.tobytes()).hexdigest()[:16], 16) % getPrime()
        
    elif getType() == "basic_covering":
          
        return int(sha1((data & LSH).tobytes()).hexdigest()[:16], 16) % getPrime()
        
    elif getType() == "l1":
        
        # data = 1 x D
        # cauchy = D x K
        # hashValues = N x K  
        # uniform = 1 x K
        # universal = K x 1
        hashValue = numpy.dot(data, LSH['cauchy']) + LSH['uniform']
        hashValue = math.floor(hashValue / getW())
        hashValue = numpy.dot(hashValue, LSH['universal']) % getPrime()
    
        return hashValue
        
    elif getType() == "l2":
        
        # data = 1 x D
        # cauchy = D x K
        # hashValues = N x K  
        # uniform = 1 x K
        # universal = K x 1
        hashValue = numpy.dot(data, LSH['gauss']) + LSH['uniform']
        hashValue = math.floor(hashValue / getW())
        hashValue = numpy.dot(hashValue, LSH['universal']) % getPrime()
    
        return hashValue
    
    elif getType() == "cosine":
        
        hashValue = numpy.dot(data, LSH['gauss'])
        hashValue[hashValue >= 0] = 1
        hashValue[hashValue < 0] = 0
        
        return int(sha1(bitarray(hashValue).tobytes()).hexdigest()[:16], 16) % getPrime()
        
#------------------------------------------------------------------------------                
"""
Build hash tables
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
	
    if candidateSizeEst + numCollision * getCPURatio() > N :
        
        #print("Use Linear Search for OutputSensitive LSH...")
        for idxPoint in range(N):        
		
            if bitdiff(query, data[idxPoint]) <= 5:
                rNN.append(idxPoint)        
        
    else:        
        
        #print("Use standard LSH for OutputSensitive LSH...")  
        candidateNN = set();

        # Remove duplicates        
        for idxTable in range(L):
            
            hashValue = hashVALUEs[idxTable]
        
            if TABLEs[idxTable][hashValue]:
                candidateNN = candidateNN.union(TABLEs[idxTable][hashValue])
        
        # Compute distance
        for idxPoint in candidateNN:
		
            if bitdiff(query, data[idxPoint]) <= 5:
                rNN.append(idxPoint)        
        
    return rNN#, candidateSizeEst, numCollision
	
#------------------------------------------------------------------------------        
"""
r-near neighbor query with LSH
"""    
def rNN_LSH(TABLEs, LSHs, query, data):    
    
    L = len(LSHs)    
    candidateNN = set()     
    
	# Removing duplicates    
    for idxTable in range(L):
        
        hashValue = hashComputation(LSHs[idxTable], query)
        
        if TABLEs[idxTable][hashValue]:
            candidateNN = candidateNN.union(TABLEs[idxTable][hashValue])    
			
    # Compute distance
    rNN = list()
    for idxPoint in candidateNN:
	
        if bitdiff(query, data[idxPoint]) <= 5:
            rNN.append(idxPoint)
        
    return rNN
#------------------------------------------------------------------------------        
"""
r-near neighbor query with Linear Search
"""    
def rNN_Linear(query, data):
    
    # Searching    
    rNN = list()    
    for idxPoint in range(getN()): 
	
        if bitdiff(query, data[idxPoint]) <= 5:
           rNN.append(idxPoint)        
           
    return rNN    

#------------------------------------------------------------------------------                    
"""
Main function for Webspam dataset with 50 queries
"""
data = pandas.read_table("C:\_Data\Dataset\_LSH\Real\webspam\webspam_data_64_bit.txt", header=None, dtype=numpy.int8)
query = pandas.read_table("C:\_Data\Dataset\_LSH\Real\webspam\webspam_query_64_bit.txt", header=None, dtype=numpy.int8)

query = query.as_matrix()
data = data.as_matrix()
numQuery = len(query)
numData = len(data)

#Conver to bit array
query_bitarray = [bitarray(query[i, :].tolist()) for i in range(numQuery)]
data_bitarray = [bitarray(data[i, :].tolist()) for i in range(numData)]

dupTime = [0 for i in range(numQuery)]
numColl = [0 for i in range(numQuery)]

candSize = [0 for i in range(numQuery)]
candSizeEst = [0 for i in range(numQuery)]

#-----------------------------------------------------------

LSHs = buildLSHsFunction()
TABLEs, HLLs = buildDataStructure(LSHs, data_bitarray)

#-----------------------------------------------------------

start = time.clock()
for i in range(numQuery):

    #rNN, candidateSize, duplicateTime = rNN_LSH(TABLEs, LSHs, query_bitarray[i], data_bitarray)
    rNN = rNN_LSH(TABLEs, LSHs, query_bitarray[i], data_bitarray)
	
    #dupTime[i] = duplicateTime * 1000000
    #candSize[i] = candidateSize
	
print("LSH Time: ", time.clock() - start, " in second \n")
#numpy.savetxt('_lsh.txt', rNN, fmt='%i', delimiter='\t')

#-----------------------------------------------------------

start = time.clock()
for i in range(numQuery):

     #rNN, candidateSizeEst, numCollision = rNN_OutLSH(TABLEs, HLLs, LSHs, query_bitarray[i], data_bitarray)
	 rNN = rNN_OutLSH(TABLEs, HLLs, LSHs, query_bitarray[i], data_bitarray)
     #candSizeEst[i] = candidateSizeEst
     #numColl[i] = numCollision
    
print("Output-sensitive LSH Time: ", time.clock() - start, " in second")

#numpy.savetxt('_dupTime.txt', dupTime, fmt='%i', delimiter='\t')
#numpy.savetxt('_numColl.txt', numColl, fmt='%i', delimiter='\t')

#numpy.savetxt('_candSize.txt', candSize, fmt='%i', delimiter='\t')
#numpy.savetxt('_candSizeEst.txt', candSizeEst, fmt='%i', delimiter='\t')

#-----------------------------------------------------------
start = time.clock()
for i in range(numQuery):

    rNN = rNN_Linear(query_bitarray[i], data_bitarray)
    
print("Linear Time: ", time.clock() - start, " in second \n")
#numpy.savetxt('_linear.txt', rNN, fmt='%i', delimiter='\t')
