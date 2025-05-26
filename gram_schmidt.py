


# # Gram-Schmidt process


# ### Matrices in Python
# Remember the structure for matrices in *numpy* is,
# ```python
# A[0, 0]  A[0, 1]  A[0, 2]  A[0, 3]
# A[1, 0]  A[1, 1]  A[1, 2]  A[1, 3]
# A[2, 0]  A[2, 1]  A[2, 2]  A[2, 3]
# A[3, 0]  A[3, 1]  A[3, 2]  A[3, 3]
# ```
# You can access the value of each element individually using,
# ```python
# A[n, m]
# ```
# You can also access a whole row at a time using,
# ```python
# A[n]
# ```
# 

# A[:, m]
# ```
# which will select the m'th column (starting at zero).
# 

# To dot product vectors u and v, use the code,
# ```python
# u @ v
# ```
# 
# All the code you should complete will be at the same level of indentation as the instruction comment.
# 



import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14 # That's 1×10⁻¹⁴ = 0.00000000000001


def gsBasis4(A) :
    B = np.array(A, dtype=np.float_) 
    B[:, 0] = B[:, 0] / la.norm(B[:, 0])
    
    B[:, 1] = B[:, 1] - B[:, 1] @ B[:, 0] * B[:, 0]
   
    if la.norm(B[:, 1]) > verySmallNumber :
        B[:, 1] = B[:, 1] / la.norm(B[:, 1])
    else :
        B[:, 1] = np.zeros_like(B[:, 1])

    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 0] * B[:, 0]
    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 1] * B[:, 1] 
    
    if la.norm(B[:, 2]) > verySmallNumber :
        B[:, 2] = B[:, 2] / la.norm(B[:, 2])
    else :
        B[:, 2] = np.zeros_like(B[:, 2])

    
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 0] * B[:, 0]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 1] * B[:, 1]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 2] * B[:, 2] 
    
    if la.norm(B[:, 3]) > verySmallNumber :
        B[:, 3] = B[:, 3] / la.norm(B[:, 3])
    else :
        B[:, 3] = np.zeros_like(B[:, 3])


    return B


def gsBasis(A):
    B = np.array(A, dtype=np.float_)
    
    for i in range(B.shape[1]):
        for j in range(i):
            proj = np.dot(B[:, i], B[:, j]) * B[:, j]
            B[:, i] = B[:, i] - proj
        
        norm = la.norm(B[:, i])
        if norm > verySmallNumber:
            B[:, i] = B[:, i] / norm
        else:
            B[:, i] = np.zeros_like(B[:, i])
    
    return B


# This function uses the Gram-schmidt process to calculate the dimension
# spanned by a list of vectors.
# Since each vector is normalised to one, or is zero,
# the sum of all the norms will be the dimension.
def dimensions(A) :
    return np.sum(la.norm(gsBasis(A), axis=0))


# ## Test your code 




A = np.array([[1,0,2,6],
              [0,1,8,2],
              [2,8,3,1],
              [1,-6,2,3]], dtype=np.float_)
gsBasis4(A)




# Once you've done Gram-Schmidt once,
# doing it again should give you the same result. Test this:
B = gsBasis4(A)
gsBasis4(A)


# In[4]:


def gsBasis(A):
    B = np.array(A, dtype=np.float_)
    
    for i in range(B.shape[1]):
        for j in range(i):
            proj = np.dot(B[:, i], B[:, j]) * B[:, j]
            B[:, i] = B[:, i] - proj
        
        norm = la.norm(B[:, i])
        if norm > verySmallNumber:
            B[:, i] = B[:, i] / norm
        else:
            B[:, i] = np.zeros_like(B[:, i])
    
    return B


def dimensions(A) :
    return np.sum(la.norm(gsBasis(A), axis=0))        


# In[5]:


# See what happens for non-square matrices
A = np.array([[3,2,3],
              [2,5,-1],
              [2,4,8],
              [12,2,1]], dtype=np.float_)
gsBasis(A)

def gsBasis(A):
    return gsBasis4(A)  # assuming gsBasis4() is the Gram-Schmidt implementation


# In[7]:


def dimensions(A):
    return np.sum(la.norm(gsBasis(A), axis=0))


# In[8]:


B = np.array([[6,2,1,7,5],
              [2,8,5,-4,1],
              [1,-6,3,2,8]], dtype=np.float_)
gsBasis(B)


# In[9]:


dimensions(B)


# In[15]:


# Now let's see what happens when we have one vector that is a linear combination of the other   
C = np.array([[1,0,2],
               [0,1,-3],
               [1,0,2]], dtype=np.float_)

gsBasis(C) 


# In[ ]:


dimensions(C)


# In[ ]:




