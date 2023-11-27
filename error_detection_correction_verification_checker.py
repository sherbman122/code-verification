# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 17:33:48 2022

@author: xzavi
"""

import vector_package
import math
import numpy

###########################################################################################
# Here we define the error operators. In this case, the error operators are the generators
# of the Lie algebra su(3)
###########################################################################################
lambda_1 = vector_package.Operator([[0,1,0], [1,0,0], [0,0,0]])
lambda_2 = vector_package.Operator([[0,0,1], [0,0,0], [1,0,0]])
lambda_3 = vector_package.Operator([[0,0,0], [0,0,1], [0,1,0]])
lambda_4 = vector_package.Operator([[0,complex(0,-1),0], [complex(0,1),0,0], [0,0,0]])
lambda_5 = vector_package.Operator([[0,0,complex(0,-1)], [0,0,0], [complex(1,0),0,0]])
lambda_6 = vector_package.Operator([[0,0,0], [0,0,complex(0, -1)], [0,complex(0,1),0]])
H_1 = vector_package.Operator([[1,0,0], [0, -1, 0] , [0,0,0]])
H_2 = vector_package.Operator([[0,0,0], [0,1,0], [0,0,-1]])


###########################################################################################
# ERROR CORRECTION: Here, we define the coefficients for the basis vectors that make up our 
# error correcting codeword, Logical 0.
###########################################################################################

# The first coefficient
a_1 = math.sqrt(10214875/168)
# The second coefficient
a_0 = math.sqrt(670371601625)
# The third coefficient
c = math.sqrt(8586854127423000)

# Here, we put the necessary data used for building symmetric vectors in a list. This
# makes it more efficient to create the symmeteric vector.

list_of_lists_correcting = [[35,1,1, a_0/c], [11,1,25, a_1/c], [11, 25,1, a_1/c], [9, 14, 14, 1/c]]

# This function generates symmetric vectors from the list above.
    
logical_zero_correcting = vector_package.generate_logical_bit(list_of_lists_correcting)

# Here, we are specifically defining the errors that we would like to correct against.
# As mentioned above, we are correcting against the infinitesimal generators of su(3).

error_space_correcting = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, H_1, H_2]

# Here, we are generating the codespace.

code_space_correcting = vector_package.Codespace(logical_zero_correcting, error_space_correcting)


# Calling this function determines whether or not the above codespace is error correcting. 
# It should print out True. Should take no longer than about 20 seconds.


print(code_space_correcting.is_error_correcting())

###########################################################################################
# ERROR DETECTION: Here, we define the coefficients for the basis vectors that make up our 
# error detecting codeword, Logical 0.
###########################################################################################

# The first coefficient:
a_0 = math.sqrt(3)/3

# Here, we put the necessary data used for building symmetric vectors in a list. This
# makes it more efficient to create the symmeteric vector.

list_of_lists_detecting = [[4,0,0, a_0], [0,2,2,1/3]]

# This function generates symmetric vectors from the list above.

logical_zero_detecting = vector_package.generate_logical_bit(list_of_lists_detecting)

# Here, we are specifically defining the errors that we would like to detect.
# As mentioned above, we are correcting against the infinitesimal generators of su(3).

error_space_detecting = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, H_1, H_2]

# Here, we are generating the codespace.

code_space_detecting = vector_package.Codespace(logical_zero_detecting, error_space_detecting)

# Calling this function determines whether or not the above codespace is error detecting. 
# It should print out True. Should take no longer than about 20 seconds.




















