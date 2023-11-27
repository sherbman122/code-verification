# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:19:56 2022

@author: xzavi
"""
import math
import numpy


###############################################################
# DEFINITION OF THE VECTOR CLASS:
###############################################################


class Vect(object):
    '''
    This class generates a vector that has the form :
    
    coefficient*< e_1^\otimes one_power, e_2^\otimes two_power, e_3^ \otimes three_power>
    
    Notes: 
    * The Vect object only represents single vectors, not a superposition of vectors.
    * Refer to the Sym_Vect class for a representation of the symmetric vectors. 
    '''
    def __init__(self, one_power, two_power, three_power, coefficient = 1):
        '''
        

        Parameters
        ----------
        one_power : Integer. The number of times e_1 ([1,0,0]) 
        appears in the tensor product
        
        two_power : Integer. The number of times e_2 ([0,1,0]) 
        appears in the tensor product
        
        three_power : Integer. The number of times e_3 ([0,0,1]) 
        appears in the tensor product
        
        coefficient : Float. The coefficient of the vector. The default is 1.

        '''
        
        
        self.one_power = one_power
        self.two_power = two_power
        self.three_power = three_power
        self.coefficient = coefficient
        # Type characterizes the tensor power. I.e. e_1 \otimes e_1 is a different type
        # than e_1 \otimes e_2. Type helps differentiate the orthogonal basis vectors.
        self.type = (one_power, two_power, three_power)
        self.dimension = one_power + two_power + three_power
        
    def get_one_power(self):
        '''
        Returns the tensor power of e_1
        '''
        return self.one_power
    
    def get_two_power(self):
        '''
        Returns the tensor power of e_2
        '''
        return self.two_power
    
    def get_three_power(self):
        '''
        Returns the tensor power of e_3
        '''
        return self.three_power
    
    def get_coefficient(self):
        '''
        Returns the tensor power of e_3
        '''
        return self.coefficient
    
    def set_coefficient(self, new_coefficient):
        '''
        new_coefficient: float or int.
        
        Changes the coefficient of the vector
        '''
        self.coefficient = new_coefficient
        
    def mult_coefficient(self, scalar):
        '''
        scalar: float or int.
        
        Multiplies the vector by the scalar.
        
        WARNING: Changes the original vector.
        '''
        old_coefficient = self.get_coefficient()
        new_coefficient = scalar*old_coefficient
        return Vect(self.get_one_power(), self.get_two_power(), self.get_three_power(), new_coefficient)
    
    def get_dimension(self):
        '''
        Retrieves the dimension of the ambient space that the vector lives in.
        '''
        return self.dimension
    
    def get_vector(self):
        '''
        Returns the original vector with a coefficient of 1.
        
        I.e. v = 4*<1,2,3> (Vect(1,2,3,4)), v.get_vector() returns 1*<1,2,3>.
        '''
        return Vect(self.one_power, self.two_power, self.three_power, 1)
    
    def __str__(self):
        '''
        Puts the vector in a readable form:
            
        v = 4*<1,2,3> (Vect(1,2,3,4))
        '''
        return str(self.coefficient) + '*<' + str(self.one_power) + ', ' + str(self.two_power) + ', ' + str(self.three_power) + '>'
    
    def get_type(self):
        '''
        Returns the type (basis tuple) of the vector.
        '''
        return self.type
    
    def is_type(self, other):
        '''
        Determines if two vectors are the same type (multiples of the same basis vector).
        '''
        if self.type == other.type:
            return True
        else:
            return False


    # NOTE: 
    # * The add function only adds vectors that are multiplies of the same basis vector.
    # * For superposition of vectors, refer to the Logical_Bit class.
   
    def __add__(self, other):
        # If the vectors that are being added together are the same "type", then adding
        # them together will result in adding the coefficients.
        if self.is_type(other):
            return Vect(self.one_power, self.two_power, self.three_power, self.coefficient + other.coefficient)
        else:
            return None
        
    # The multiplication operator * is understood to be the standard dot product
        
    def __mul__(self, other):
        if self.is_type(other):
            return self.coefficient*other.coefficient
        else:
            return 0
    
    # Two vectors are understood to be the same if 
    # their coefficient and types are the same.        

    def __eq__(self, other):
        if self.is_type(other) and self.get_coefficient() == other.get_coefficient():
            return True
        else:
            return False
        
    # This is the first iteration of applying the X (bit-flip) operator
    # For a generalized way of applying operators to vectors, refer to the applyOperator
    # function.
        
    def applyX(self):
        '''
        This operation permutes the tensor powers in the following way,
        
        one_power ----> two_power
        
        two_power ----> three_power
        
        three_power ---> one_power
        
        
        '''
        return Vect(self.three_power, self.one_power, self.two_power, self.coefficient)
        
        
        
##############################################################
# DEFINITION OF THE SYMMETRIC VECTOR CLASS
##############################################################        
        
class Sym_Vect(Vect):
    '''
    The Sym_Vect class corresponds to the symmetric vectors. Everything for this class is 
    about the same as a standard vector. The main difference is that the dot product is 
    adjusted to account for multiplicity, and the representation of a symmetric vector 
    utilizes square brackets "[,]" instead of angled brackets "<,>". 
    '''
    def __init__(self, one_power, two_power, three_power, coefficient = 1):
        Vect.__init__(self, one_power, two_power, three_power, coefficient)
        # This corresponds to the multiplicity coefficient. To calculate it,
        # we use the orbit stabilizer theorem:
        # permutation coefficient = number of letters being acted upon (a.k.a dimension)/ stabilizer of the vector
        self.permutation_coefficient = math.factorial(self.dimension)/self.stabilizer_order()
    
    
    def get_permutation_coefficient(self):
        '''
        Returns the multiplicity coefficient of the symmetric vector.
        '''
        return self.permutation_coefficient
    
    def __str__(self):
        return str(self.coefficient) + '*[' + str(self.one_power) + ', ' + str(self.two_power) + ', ' + str(self.three_power) + ']'
    
    def stabilizer_order(self):
        '''
        Calculates the stabilizer of a vector via the orbit stabilizer theorem.

        '''
        one_power = self.get_one_power()
        two_power = self.get_two_power()
        three_power = self.get_three_power()
        stabilizer_order = math.factorial(one_power)*math.factorial(two_power)*math.factorial(three_power)
        return stabilizer_order
    
    # Multiplication * is understood to be the dot product. Because these vectors represent
    # a class of vectors, we have to account for this by multiplying by the multiplicity
    # constant.
    
    def __mul__(self, vector_two):
        if self.is_type(vector_two):
            return self.permutation_coefficient*self.get_coefficient()*vector_two.get_coefficient()
        else:
            return 0
    # NOTE: 
    # * The add function only adds vectors that are multiplies of the same basis vector.
    # * For superposition of vectors, refer to the Logical_Bit class.  
    
    def __add__(self, other):
        if self.is_type(other):
            return Sym_Vect(self.one_power, self.two_power, self.three_power, self.coefficient + other.coefficient)
        else:
            return None
        
    def mult_coefficient(self, scalar):
        '''
        scalar: float or int.
        
        Multiplies the vector by the scalar.
        
        WARNING: Changes the original vector.
        '''
        old_coefficient = self.get_coefficient()
        new_coefficient = scalar*old_coefficient
        return Sym_Vect(self.get_one_power(), self.get_two_power(), self.get_three_power(), new_coefficient)
        
    def applyX(self):
        return Sym_Vect(self.three_power, self.one_power, self.two_power, self.coefficient)

##############################################################
# DEFINITION OF THE LOGICAL BIT CLASS
##############################################################




class Logical_Bit(object):
    def __init__(self, vect_objects, state = None):
        '''
        The Logical_Bit object can be thought of as the superposition of (symmetric) vectors.
        It is represented internally as a dictionary object, but it prints out as a superposition
        of vectors. 

        Parameters
        ----------
        vect_objects : List of (Sym)_Vect objects. These will make up the superposition of
        vectors.
        state : int, 0,1, or 2. This represents if a bit is a logical zero, one, or two bit.
            The default is None.

        '''
        # One can specify the state; however, the state attribute may be useless
        self.state = state
        self.vect_objects = vect_objects
        
        # This process records the vectors that are present in the logical bit
        eigenspace = {}
        for vector in vect_objects:
            # This checks whether or not the basis vector is already in the dictionary
            # If the basis vector is not already in the dictionary, the basis vector is recorded
            # and the actual vector is taken to be the value.
            basis_tuple = vector.get_type()
            if basis_tuple not in eigenspace:
                eigenspace[basis_tuple] = vector
            # If the basis vector is already in the eigenspace, then the coefficient
            # of the vector is adjusted accordingly.
            elif basis_tuple in eigenspace:
                eigenspace[basis_tuple] += vector
                
                
        self.eigenspace = eigenspace
    
    def get_state(self):
        '''
        Returns the state of the bit.
        '''
        return self.state
    
    def get_eigenspace(self):
        '''
        Returns the eigenspace of the bit.
        '''
        return self.eigenspace
    
    def get_eigenspace_readable(self):
        '''
        Returns a readable version of the bit's eigenspace.
        '''
        readable_eigenspace = {}
        for vector in self.eigenspace:
            readable_eigenspace[vector] = str(self.eigenspace[vector])
        return readable_eigenspace
    
    def get_vect_objects(self):
        '''
        Returns the list of vector objects that the user inputted.
        '''
        return self.vect_objects
    
    def set_state(self, name_change):
        '''
        Changes the state of the logical bit. (May be useful for when one applies 
                                               operators to a bit)
        '''
        self.state = name_change
        
    def insert_eigenstate(self, eigenstate):
        '''
        eigenstate: Vect or Sym_Vect object. 
        
        Inserts the specified eigenstate into the eigenspace dictionary. If the eigenstate
        is already in the dictionary, then it will adjust the coefficient of the eigenstate
        already present in the eigenspace dictionary.
        
        WARNING: This changes the eigenspace of the original logical bit.
        '''
        if eigenstate.get_type() in self.eigenspace:
            self.eigenspace[eigenstate.get_type()] += eigenstate
            self.vect_objects.append(eigenstate)
        else:
            self.eigenspace[eigenstate.get_type()] = eigenstate
            self.vect_objects.append(eigenstate)
            
    def symmetrize(self):
        '''
        Turns a logical bit into a symmetric logical bit.
        '''
        
        new_vect_objects = self.get_vect_objects()
        ans_list = []
        for vector in new_vect_objects:
            symmetrization = Vect_to_Sym_Vect(vector)
            ans_list.append(symmetrization)
        ans_bit = Logical_Bit(ans_list)
        return ans_bit
            
            
    def __add__(self, other):
        self_objects = self.get_vect_objects()
        other_objects = other.get_vect_objects()
        new_objects = self_objects + other_objects
        return Logical_Bit(new_objects)
    
# Multiplication is understood to be the dot product of a superposition of vectors.
            
    def __mul__(self, other):
        final_sum = 0
        if type(other) == type(Vect(1,2,3,4)) or type(other) == type(Sym_Vect(1,2,3,4)):
            other = Logical_Bit([other])
        for vector_type in self.get_eigenspace():
            for other_vector_type in other.get_eigenspace():
                vector_one = other.get_eigenspace()[other_vector_type]
                vector_two = self.get_eigenspace()[vector_type]
                final_sum += vector_one*vector_two
        return final_sum
    
    def scalar_mult(self, other):
        eigenspace = self.get_eigenspace()
        vect_objects = []
        for vector_type in eigenspace:
            scaled_vector = eigenspace[vector_type].mult_coefficient(other)
            vect_objects.append(scaled_vector)
        return Logical_Bit(vect_objects)
        
    
    def __str__(self):
        '''
        Returns the superposition of all the vectors that are in the eigenspace.
        '''
        result_string = ''
        j = 1
        for vector in self.get_eigenspace().keys():
            if j == len(self.get_eigenspace().keys()):
                    result_string += str(self.get_eigenspace()[vector])
            else:
                j += 1
                result_string += str(self.get_eigenspace()[vector]) + ' + '
        return result_string
    
    def applyX(self):
        '''
        Applies the X operator to every vector in the eigenspace.
        '''
        result = []
        for vector in self.eigenspace :
            Xvector = self.eigenspace[vector].applyX()
            result.append(Xvector)
        return Logical_Bit(result)
    
    
    
class Operator(object):
    '''
    Creates an object that is basically a matrix; however, there is greater ability
    to manipulate it.
    '''
    def __init__(self, list_of_entries):
        '''
        list_of_entires: A three item list. Each item is a list with 3 entries.
        
        Creates a three by three matrix.
        '''
        self.list_of_entries = list_of_entries
        self.matrix = numpy.matrix(list_of_entries)
        
    def get_entries(self):
        '''
        Returns the list of entries of the operator. 
        '''
        return self.list_of_entries
        
    def get_matrix(self):
        '''
        Returns the matrix of the operator.
        '''
        return self.matrix
    
    def transpose(self):
        return matrix_to_operator(self.matrix.transpose())
    
        
    def __str__(self):
        '''
        Returns a matrix representation of the operator.
        '''
        return str(self.matrix)
    

# Operator multiplication is inherited from matrix multiplication.


    def __mul__(self, other):
        other_new = other
        self_new = self
        if type(self) == type(Operator([[0,0,1], [0,0,1], [0,0,1]])):
            self_new = self.get_matrix()
        if type(other) == type(Operator([[0,0,1], [0,0,1], [0,0,1]])):      
            other_new = other.get_matrix()
        return matrix_to_operator(self_new*other_new)
    
    
        
    
# Operator addition is inherited from matrix multiplication.
    
    def __add__(self, other):
        if type(self) == type(Operator([[0,0,1], [0,0,1], [0,0,1]])):
            self_new = self.get_matrix()
        if type(other) == type(Operator([[0,0,1], [0,0,1], [0,0,1]])):      
            other_new = other.get_matrix()
        return matrix_to_operator(self_new + other_new)
    
    def __pow__(self, other):
        return matrix_to_operator(self.get_matrix()**other)
    
    
    
class Lie_generator(Operator):
    
    def __init__(self, list_of_entries):
        Operator.__init__(self, list_of_entries)
        
    def __mul__(self, other):
        if type(other) == type(Vect(1,0,0)) or type(other) == type(Sym_Vect(1,0,0)):
            return applyOperator_Lie_algebra(self, other)
        elif type(other) == type(Logical_Bit([Vect(1,0,0)])):
            return applyOperator_Lie_algebra_logical_bit(self, other)
        elif type(other) == type(Operator([[1,2,3], [4,5,6], [7,8,9]])):
            return Operator.__mul__(self, other)
        
class Lie_group_element(Operator):
    
    def __init__(self, list_of_entries):
        Operator.__init__(self, list_of_entries)
        
    def __mul__(self, other):
        if type(other) == type(Vect(1,0,0)) or type(other) == type(Sym_Vect(1,0,0)):
            return applyOperator_vector(self, other)
        elif type(other) == type(Logical_Bit([Vect(1,0,0)])):
            return applyOperator_logical_bit(self, other)
        else:
            return Operator.__mul__(self, other)
        
        def transpose(self):
            p = matrix_to_operator(self.matrix.transpose())
            return Lie_converter(p)
        
        
def Lie_converter(operator_one):
    operator_one_matrix = operator_one.get_entries()
    v = Lie_generator(operator_one_matrix)
    return v
        
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    
def matrix_to_operator(matrix):
    '''
    Converts a matrix to an operator.
    
    Returns an operator object.
    '''
    ans = []
    i = 0
    while i <9:
        L = [matrix.item(i), matrix.item(i+1), matrix.item(i+2)]
        ans.append(L)
        i +=3 
    return Operator(ans)



###########################################################
# In this section, we will work on building the error operators and the representations
# of the error operators
#########################################################

# Originally, the following function was created to change a vector that was in tensor
# form to a matrix. I wanted to do this so that I could do matrix multiplication. I realized
# very quickly that programing the kronecker product of matrices may not be feasible. So,
# instead, I decided to take a more natural approach to the action of an operator on the 
# tensor product structure, which is detailed below.

def vector_to_matrix(vector):
    '''
    vector: numpy.matrix one-dimensional object.
    
    Converts a vector (object) to a one-dimensional array for matrix multiplication.
    
    NOTE: This function is still under construction becasue I found a way
    to apply operators to a vector that has a tensor product structure. See
    the Applying Operators section.
    '''

    one_power = vector.get_one_power()
    two_power = vector.get_two_power()
    three_power = vector.get_three_power()
    
    ans = []
    
    for i in range(one_power):
        a = [1]
        b = [0]
        c = [0]
        ans.append(a)
        ans.append(b)
        ans.append(c)
    
    for k in range(two_power):
        a = [0]
        b = [1]
        c = [0]
        ans.append(a)
        ans.append(b)
        ans.append(c)
        
    for j in range(three_power):
        a = [0]
        b = [0]
        c = [1]
        ans.append(a)
        ans.append(b)
        ans.append(c)
    
    return numpy.matrix(ans)

def matrix_to_vector_basis(matrix):
    '''
    Converts an n by 1 basis vector (numpy.matrix object) to a Vect object.
    
    Outline of process: This function takes advantage of orthogonal projection.
    Because the input vector is considered a basis vector, orthogonal projection indicates
    what vector subspace the vector lies in.
    
    Returns a vector object that has the same form as the original matrix.
    '''
    
    e_1_projector = numpy.matrix([[1,0,0]])
    e_2_projector = numpy.matrix([[0,1,0]])
    e_3_projector = numpy.matrix([[0,0,1]])
    
    if e_1_projector*matrix != 0:
        # Here, we have to use the int command because otherwise the product is a matrix.
        coefficient = int(e_1_projector*matrix) 
        return Vect(1,0,0,coefficient)
    elif e_2_projector*matrix != 0:
        coefficient = int(e_2_projector*matrix)
        return Vect(0,1,0,coefficient)
    else:
        coefficient = int(e_3_projector*matrix)
        return Vect(0,0,1, coefficient)


#######################################################################
# APPLYING OPERATORS
#######################################################################


# In this section, I define a bunch of intermediate functions to use in the 
# final function, which is applying a generic operator to a generic vector.
# I do this essentially by using the linearity of matrices.


# This first function applies a specified operator to the basis vectors of R^3

def applyOperator_Basis(operator):
    '''
    operator: an operator object.
    
    Returns a dictionary such that the keys are the basis tuples (vectors) and the value
    of a key is the value of the operator acting on the basis tuple. This construction is
    very similar to the graph construction where the key is x and the value is f(x).
    
    For example, if X is the operator, the output is the dictionary:
        
    {e_1: Xe_1, e_2 : Xe_2, e_3: X e_3}
    
    
    '''
    
    # This step converts the basis Vect objects to numpy.matrix objects
    e_1 = vector_to_matrix(Vect(1,0,0))
    e_2 = vector_to_matrix(Vect(0,1,0))
    e_3 = vector_to_matrix(Vect(0,0,1))
    list_of_basis_vectors = [e_1, e_2, e_3]
    ans_dict = {}
    # This step retrieves the matrix representation of the operator
    operator = operator.get_matrix()
    # This step applies the operator to each basis vector
    for vector in list_of_basis_vectors:
        # This step calculates the matrix action on the vector
        new_vector = operator*vector
        # This step records the output of the operation
        ans_dict[str(matrix_to_vector_basis(vector))] = new_vector
    return ans_dict

    



def projection_operation(determiner):
    '''
    determiner: numpy.matrix object ( a 3 by 1 matrix).
    
    This function does something very similar to the matirx to vector function. It uses
    orthogonal pojection to deterine the subspace that the "determiner" is in.
    
    The function returns a tuple in the form:
        
    ( the coefficient of the vector, the vector subspace of R^3 the vector is in )
    '''
    
    if numpy.matrix([[1,0,0]])*determiner != 0:
        coefficient_one = complex(numpy.matrix([[1,0,0]])*determiner)
        vector_space = Vect(1,0,0)
    elif numpy.matrix([[0,1,0]])*determiner != 0:
        coefficient_one = complex(numpy.matrix([[0,1,0]])*determiner)
        vector_space = Vect(0,1,0)
    elif numpy.matrix([[0,0,1]])*determiner != 0:
        coefficient_one = complex(numpy.matrix([[0,0,1]])*determiner)
        vector_space = Vect(0,0,1)
    else:
        coefficient_one = 0
        vector_space = Vect(0,0,0, 0)
    return (coefficient_one, vector_space)
    
    

def applyOperator_vector(operator, vector):
    '''
    Applys the operator to the vector, systematically, of course. (Using the standard representation)
    
    NOTE: This only works for single vectors currently, and I am working on extending this
    to logical bit objects (i.e. a superpostion of vectors).
    
    ANOTHER IMPORTANT NOTE: This process only works for operators that preserve the basis
    eigenspaces. Because we are relying on projection, if the operator were to mix the eigenspaces,
    the projection process will not work properly. I have to make adjustments for this, but for the
    project, the error operators do not mix eigenspaces, they only swap them.
    '''
    
    # First, we initialize a matrix representation of the basis vectors
    e_1 = vector_to_matrix(Vect(1,0,0))
    e_2 = vector_to_matrix(Vect(0,1,0))
    e_3 = vector_to_matrix(Vect(0,0,1))
    
    
    
    # Next, we apply the operator to the basis vectors
    image_of_operator_dict = applyOperator_Basis(operator)
    
    # Next, we initialize the variables that will be used to determine the tensor structure
    # (type) of the final vector.
    (new_one_power, new_two_power, new_three_power) = (0,0,0)
    
    # This process determines where the e_1 vector gets sent to. We need this process because
    # we can visually determine what subspace the e_1 vector gets sent to, but the computer
    # can not tell very well. 
    
    
    # Step one is to retrieve the image of the e_1 vector under the operator. i.e operator(e_1)
    one_determiner = image_of_operator_dict[str(matrix_to_vector_basis(e_1))]
    # Step two is to determine which basis eigenspace the image of e_1 lands in.
    # Remember, we are making the very big assumption that the operators do not map
    # vectors to a superposition of basis vectors. 
    (new_coefficient_one, new_vector_space) = projection_operation(one_determiner)
    # Step three is to assign the vector tensor power of e_1 to the vector tensor power of the
    # image of e_1.
    if new_vector_space == Vect(1,0,0):
        new_one_power = vector.get_one_power()
    elif new_vector_space == Vect(0,1,0):
        new_two_power = vector.get_one_power()
    elif new_vector_space == Vect(0,0,1):
        new_three_power = vector.get_one_power()
        
    # Next, we repeat the same process for the other two basis vectors.
    
    two_determiner = image_of_operator_dict[str(matrix_to_vector_basis(e_2))]
    (new_coefficient_two, new_vector_space) = projection_operation(two_determiner)
    if new_vector_space == Vect(1,0,0):
        new_one_power = vector.get_two_power()
    elif new_vector_space == Vect(0,1,0):
        new_two_power = vector.get_two_power()
    elif new_vector_space == Vect(0,0,1):
        new_three_power = vector.get_two_power()
        
    three_determiner = image_of_operator_dict[str(matrix_to_vector_basis(e_3))]
    (new_coefficient_three, new_vector_space) = projection_operation(three_determiner)
    if new_vector_space == Vect(1,0,0):
        new_one_power = vector.get_three_power()
    elif new_vector_space == Vect(0,1,0):
        new_two_power = vector.get_three_power()
    elif new_vector_space == Vect(0,0,1):
        new_three_power = vector.get_three_power()
        
    # This step determines the final coefficient for the resulting vector
    
    final_coefficient = vector.get_coefficient()*(new_coefficient_one**vector.get_one_power())*(new_coefficient_two**vector.get_two_power())*(new_coefficient_three**vector.get_three_power())
    
    if type(vector) == type(Sym_Vect(1,1,1)):
        return Sym_Vect(new_one_power, new_two_power, new_three_power, final_coefficient)
    else:
        return Vect(new_one_power, new_two_power, new_three_power, final_coefficient)


def Vect_to_Sym_Vect(vector):
    '''
    Converts a vector to a symmetric vector
    '''
    return Sym_Vect(vector.get_one_power(), vector.get_two_power(), vector.get_three_power(), vector.get_coefficient())
        
    
    

def applyOperator_logical_bit(operator, logical_bit):
    ans_bit = Logical_Bit([])
    eigenspace = logical_bit.get_eigenspace()
    for vector_type in eigenspace:
        new_vector = eigenspace[vector_type]
        new_image_vector = applyOperator_vector(operator, new_vector)
        ans_bit.insert_eigenstate(new_image_vector)
    return ans_bit










def applyOperator_tensor_product_vector(list_of_operators, vector):
    '''
    This makes it so that we can apply different operators to different tensors.
    i.e. H\otimes I \otimes H
    '''
    one_power = vector.get_one_power()
    two_power = vector.get_two_power()
    three_power = vector.get_three_power()
    coefficient = vector.get_coefficient()
    #ans_bit = []
    operator_length = len(list_of_operators)
    if one_power+two_power+three_power  == operator_length:
        i = 0
        (new_one_power, new_two_power, new_three_power, new_coefficient) = (one_power, two_power, three_power, 1)
        while i < one_power:
            basis_vector = Vect(1,0,0, coefficient)
            image_vector = applyOperator_vector(list_of_operators[i], basis_vector)
            if image_vector.get_coefficient() != 0 or not image_vector.is_type(Vect(0,0,0,0)):
                if image_vector.is_type(basis_vector):
                    (new_one_power, new_two_power, new_three_power) = (new_one_power, new_two_power, new_three_power)
                    new_basis_vector = Vect(1,0,0)
                elif new_one_power >= 1:
                    if image_vector.is_type(Vect(0,1,0, coefficient)):
                        (new_one_power, new_two_power, new_three_power) = (new_one_power - 1, new_two_power + 1,new_three_power)
                        new_basis_vector = Vect(0,1,0)
                    else:
                        (new_one_power, new_two_power, new_three_power) = (new_one_power - 1, new_two_power,new_three_power +1)
                        new_basis_vector = Vect(0,0,1)
                else:
                    return Vect(0,0,0,0)
                new_coefficient *= new_basis_vector*image_vector/abs(new_basis_vector*image_vector)
                #new_image_vector = Vect(new_one_power, new_two_power, new_three_power, new_coefficient*coefficient)
                #ans_bit.append(new_image_vector)
                i +=1
            else:
                return Vect(0,0,0,0)
                i +=1
            
            
        while i < one_power + two_power:
            basis_vector = Vect(0,1,0, coefficient)
            image_vector = applyOperator_vector(list_of_operators[i], basis_vector)
            if image_vector.get_coefficient() != 0 or not image_vector.is_type(Vect(0,0,0,0)):
                if image_vector.is_type(basis_vector):
                    (new_one_power, new_two_power, new_three_power) = (new_one_power, new_two_power, new_three_power)
                    new_basis_vector = Vect(0,1,0)
                elif new_two_power >=1:
                    if image_vector.is_type(Vect(1,0,0, coefficient)):
                        (new_one_power, new_two_power, new_three_power) = (new_one_power + 1, new_two_power - 1,new_three_power)
                        new_basis_vector = Vect(1,0,0)  
                    else:
                        (new_one_power, new_two_power, new_three_power) = (new_one_power, new_two_power -1, new_three_power +1)
                        new_basis_vector = Vect(0,0,1)
                else:
                    return Vect(0,0,0,0)
                new_coefficient *= new_basis_vector*image_vector/abs(new_basis_vector*image_vector)
                #new_image_vector = Vect(new_one_power, new_two_power, new_three_power, new_coefficient*coefficient)
                #ans_bit.append(new_image_vector)
                i +=1
            else:
                return Vect(0,0,0,0)
                i +=1
            
            
        while i < one_power + two_power + three_power:
            basis_vector = Vect(0,0,1, coefficient)
            image_vector = applyOperator_vector(list_of_operators[i], basis_vector)
            if image_vector.get_coefficient() != 0 or not image_vector.is_type(Vect(0,0,0,0)):
                if image_vector.is_type(basis_vector):
                    (new_one_power, new_two_power, new_three_power) = (new_one_power, new_two_power, new_three_power)
                    new_basis_vector = Vect(0,0,1)
                elif new_three_power >= 1:
                    if image_vector.is_type(Vect(1,0,0, coefficient)):
                        (new_one_power, new_two_power, new_three_power) = (new_one_power + 1, new_two_power, new_three_power - 1)
                        new_basis_vector = Vect(1,0,0)
                    else:
                        (new_one_power, new_two_power, new_three_power) = (new_one_power, new_two_power + 1, new_three_power -1)
                        new_basis_vector = Vect(0,1,0)
                else:
                    return Vect(0,0,0,0)
                new_coefficient *= new_basis_vector*image_vector/abs(new_basis_vector*image_vector)
                #new_image_vector = Vect(new_one_power, new_two_power, new_three_power, new_coefficient*coefficient)
                #ans_bit.append(new_image_vector)
                i +=1
            else:
                return Vect(0,0,0,0)
                i += 1
            
            
    else:
        return None
    
    
    ans_vector = Vect(new_one_power, new_two_power, new_three_power, coefficient*new_coefficient)
    
    if type(vector) == type(Sym_Vect(1,0,0)):
        return Vect_to_Sym_Vect(ans_vector)
    
    return ans_vector

def applyOperator_selective(operator, tensor_power, vector):
    if tensor_power == 0:
        return Vect(0,0,0,0)
    new_one_power = vector.get_one_power()
    new_two_power = vector.get_two_power()
    new_three_power = vector.get_three_power()
    coefficient = vector.get_coefficient()
    new_coefficient = 1
    if tensor_power <= new_one_power:
        basis_vector = Vect(1,0,0, coefficient)
        image_vector = applyOperator_vector(operator, basis_vector)
        if image_vector.get_coefficient() != 0 or not image_vector.is_type(Vect(0,0,0,0)):
            if image_vector.is_type(basis_vector):
                (new_one_power, new_two_power, new_three_power) = (new_one_power, new_two_power, new_three_power)
                new_basis_vector = Vect(1,0,0)
            elif new_one_power >= 1:
                if image_vector.is_type(Vect(0,1,0, coefficient)):
                    (new_one_power, new_two_power, new_three_power) = (new_one_power - 1, new_two_power + 1,new_three_power)
                    new_basis_vector = Vect(0,1,0)
                else:
                    (new_one_power, new_two_power, new_three_power) = (new_one_power - 1, new_two_power,new_three_power +1)
                    new_basis_vector = Vect(0,0,1)
            else:
                return Vect(0,0,0,0)
            new_coefficient *= new_basis_vector*image_vector/abs(new_basis_vector*image_vector)
        else:
            return Vect(0,0,0,0)
        
    elif new_one_power < tensor_power <= new_one_power + new_two_power:
        basis_vector = Vect(0,1,0, coefficient)
        image_vector = applyOperator_vector(operator, basis_vector)
        if image_vector.get_coefficient() != 0 or not image_vector.is_type(Vect(0,0,0,0)):
            if image_vector.is_type(basis_vector):
                (new_one_power, new_two_power, new_three_power) = (new_one_power, new_two_power, new_three_power)
                new_basis_vector = Vect(0,1,0)
            elif new_two_power >=1:
                if image_vector.is_type(Vect(1,0,0, coefficient)):
                    (new_one_power, new_two_power, new_three_power) = (new_one_power + 1, new_two_power - 1,new_three_power)
                    new_basis_vector = Vect(1,0,0)  
                else:
                    (new_one_power, new_two_power, new_three_power) = (new_one_power, new_two_power -1, new_three_power +1)
                    new_basis_vector = Vect(0,0,1)
            else:
                return Vect(0,0,0,0)
            new_coefficient *= new_basis_vector*image_vector/abs(new_basis_vector*image_vector)
        else:
            return Vect(0,0,0,0)
        
    else:
        basis_vector = Vect(0,0,1, coefficient)
        image_vector = applyOperator_vector(operator, basis_vector)
        if image_vector.get_coefficient() != 0 or not image_vector.is_type(Vect(0,0,0,0)):
            if image_vector.is_type(basis_vector):
                (new_one_power, new_two_power, new_three_power) = (new_one_power, new_two_power, new_three_power)
                new_basis_vector = Vect(0,0,1)
            elif new_three_power >= 1:
                if image_vector.is_type(Vect(1,0,0, coefficient)):
                    (new_one_power, new_two_power, new_three_power) = (new_one_power + 1, new_two_power, new_three_power - 1)
                    new_basis_vector = Vect(1,0,0)
                else:
                    (new_one_power, new_two_power, new_three_power) = (new_one_power, new_two_power + 1, new_three_power -1)
                    new_basis_vector = Vect(0,1,0)
            else:
                return Vect(0,0,0,0)
            new_coefficient *= new_basis_vector*image_vector/abs(new_basis_vector*image_vector)
        else:
            return Vect(0,0,0,0)
        
    ans_vector = Vect(new_one_power, new_two_power, new_three_power, coefficient*new_coefficient)
        
    if type(vector) == type(Sym_Vect(1,0,0)):
        return Vect_to_Sym_Vect(ans_vector)
        
    return ans_vector

        












def applyOperator_Lie_algebra(operator, vector):
    '''
    This takes in a vector and applies the operator if it is a lie generator.
    Takes into account representation, but it will only be for (p,0) representations.
    '''
    dimension = vector.get_dimension()
    I = Operator([[1,0,0], [0,1,0], [0,0,1]])
    list_of_vectors = []
    for i in range(dimension):
        list_of_operators = []
        #list_of_strings = []
        for j in range(dimension):
            if j == i:
                list_of_operators.append(operator)
                #list_of_strings.append('operator')
            else:
                list_of_operators.append(I)
                #list_of_strings.append('I')
        #print(list_of_strings)
        new_vector = applyOperator_tensor_product_vector(list_of_operators, vector)
        list_of_vectors.append(new_vector)
        
    ans_bit = Logical_Bit(list_of_vectors)
    return ans_bit



def applyH_one_fast_tensor(vector):
    difference = vector.get_one_power() - vector.get_two_power
    if type(vector) == type(Sym_Vect(1,0,0)):
        return Sym_Vect(vector.get_one_power(), vector.get_two_power(), vector.get_three_power(), vector.get_coefficient()*difference)
    else:
        return Vect(vector.get_one_power(), vector.get_two_power(), vector.get_three_power(), vector.get_coefficient()*difference)
    
    
    
    
def applyOperator_Lie_algebra_fast(operator, vector):
    '''
    This takes in a vector and applies the operator if it is a lie generator.
    Takes into account representation, but it will only be for (p,0) representations.
    '''
    one_power = vector.get_one_power()
    two_power = vector.get_two_power()
    three_power = vector.get_three_power()
    coefficient = vector.get_coefficient()
    previous_power = 0
    
    list_of_vectors = []
    for i in [one_power, two_power + one_power, three_power +one_power + two_power]:
        new_vector = applyOperator_selective(operator, i, vector)
        new_coefficient = new_vector.get_coefficient()
        new_one_power, new_two_power, new_three_power = (new_vector.get_one_power(), new_vector.get_two_power(), new_vector.get_three_power())
        if previous_power != i:
            if new_one_power < one_power:
                if new_two_power > two_power:
                    new_coefficient = new_two_power*coefficient
                elif new_three_power > three_power:
                    new_coefficient = new_three_power*coefficient
            elif new_two_power < two_power:
                if new_one_power > one_power:
                    new_coefficient = new_one_power*coefficient
                elif new_three_power > three_power:
                    new_coefficient = new_three_power*coefficient
            elif new_three_power < three_power:
                if new_one_power > one_power:
                    new_coefficient = new_one_power*coefficient
                elif new_two_power > two_power:
                    new_coefficient = new_two_power*coefficient
            else:
                if i == one_power:
                    new_coefficient *= one_power
                elif i == one_power + two_power:
                    new_coefficient *= two_power
                else:
                    new_coefficient *= three_power
                
            new_vector = Sym_Vect(new_one_power, new_two_power, new_three_power, new_coefficient)
            list_of_vectors.append(new_vector)
            previous_power = i
        
        
        
        
        
        
    ans_bit = Logical_Bit(list_of_vectors)
    return ans_bit



def applyOperator_Lie_algebra_logical_bit_fast(operator, logical_bit):
    
    if type(logical_bit) == type(Vect(1,0,0)) or type(logical_bit) == type(Sym_Vect(1,0,0)):
        logical_bit = Logical_Bit([logical_bit])
    
    eigenspace = logical_bit.get_eigenspace()
    image_bits_list = []
    for vector in eigenspace.values():
        image_bit = applyOperator_Lie_algebra_fast(operator, vector)
        image_bits_list.append(image_bit)
    
    image_vector_list = []
    
    for bit in image_bits_list:
        bit_eigenspace = bit.get_eigenspace()
        for image_vector in bit_eigenspace.values():
            image_vector_list.append(image_vector)
            
    ans_bit = Logical_Bit(image_vector_list)
    return ans_bit
    
    

        
    


def applyOperator_Lie_algebra_logical_bit(operator, logical_bit):
    
    eigenspace = logical_bit.get_eigenspace()
    image_bits_list = []
    for vector in eigenspace.values():
        image_bit = applyOperator_Lie_algebra(operator, vector)
        image_bits_list.append(image_bit)
    
    image_vector_list = []
    
    for bit in image_bits_list:
        bit_eigenspace = bit.get_eigenspace()
        for image_vector in bit_eigenspace.values():
            image_vector_list.append(image_vector)
            
    ans_bit = Logical_Bit(image_vector_list)
    return ans_bit



def inner_product_vector(vector_one, vector_two):
    return vector_one*vector_two

def inner_product_logical_bit(logical_bit_one, logical_bit_two):
    return logical_bit_one*logical_bit_two

def normalize_logical_bit(logical_bit_one):
    magnitude_square = inner_product_logical_bit(logical_bit_one, logical_bit_one)
    magnitude = math.sqrt(magnitude_square)
    return logical_bit_one.scalar_mult(1/magnitude)



def expectation_value_standard(operator, vector_one, vector_two):
    '''
    <vector_two| operator |vector_one>
    '''
    if type(vector_one) == type(Vect(1,0,0)) or type(vector_one) == type(Sym_Vect(1,0,0)):
        image_vector = applyOperator_vector(operator, vector_one)
    else:
        image_vector = applyOperator_logical_bit(operator, vector_one)
    return vector_two*image_vector

def expectation_value_tensor(operator, vector_one, vector_two):
    '''
    <vector_two| operator |vector_one>
    '''
    if type(vector_two) == type(Vect(1,0,0)) or type(vector_two) == type(Sym_Vect(1,0,0)):
        image_vector = applyOperator_Lie_algebra_fast(operator, vector_one)
        new_vector = Logical_Bit([vector_two])
    else:
        if type(vector_one) == type(Vect(1,0,0)) or type(vector_one) == type(Sym_Vect(1,0,0)):
            vector_one = Logical_Bit([vector_one])
        image_vector = applyOperator_Lie_algebra_logical_bit_fast(operator, vector_one)
        new_vector = vector_two
    return inner_product_logical_bit(image_vector, new_vector)



def generate_square(list_one, list_two):
    ans_list = []
    for element_one in list_one:
        for element_two in list_two:
            v = (element_one, element_two)
            ans_list.append(v)
    return ans_list

def generate_logical_bit(list_of_list, Sym_vect = True):
    vect_objects = []
    for lists in list_of_list:
        [one_power, two_power, three_power] = [lists[0], lists[1], lists[2]]
        if len(lists) == 4:
            coefficient = lists[3]
        else:
            coefficient = 1
        if Sym_vect:
            new_vector = Sym_Vect(one_power, two_power, three_power, coefficient)
        else:
            new_vector = Vect(one_power, two_power, three_power, coefficient)
        vect_objects.append(new_vector)
    return Logical_Bit(vect_objects)
        




class Codespace (object):
    def __init__ (self, logical_zero, error_space = None):
        X = Lie_group_element([[0,0,1], [1,0,0], [0,1,0]])
        self.logical_zero = normalize_logical_bit(logical_zero)
        self.logical_one = X*self.logical_zero
        self.logical_two = X*self.logical_one
        self.error_space = error_space
        self.error_detection = error_space
        I = Operator([[1,0,0], [0,1,0], [0,0,1]])
        if I not in self.error_space:
            self.error_space.append(I)
        error_correction =[]
        for error_one in self.error_space:
            error_one = error_one.transpose()
            for error_two in self.error_space:
                v = error_one*error_two
                error_correction.append(v)
        self.error_correction = error_correction
        
    def get_error_detection(self):
        return self.error_detection
    
    def get_error_correction(self):
        return self.error_correction
    
    def get_logical_zero(self):
        return self.logical_zero
    
    def get_logical_one(self):
        return self.logical_one
    
    def get_logical_two(self):
        return self.logical_two
    
    def get_error_space(self):
        return self.error_space
    
    def set_error_space(self, new_errors):
        self.error_space = new_errors
        
    def insert_error(self, new_error):
        self.error_space.append(new_error)
        
    def remove_error(self, error):
        ans_space = []
        if error in self.get_error_space():
            for non_error in self.get_error_space():
                if non_error != error:
                    ans_space.append(non_error)
        self.set_error_space(ans_space)
            

            
        
    def is_error_detecting(self):
        error_detecting = True
        error_detection = self.get_error_detection()
        (logical_zero, logical_one, logical_two) = (self.get_logical_zero(), self.get_logical_one(), self.get_logical_two())
        conditions_matrix = generate_square([logical_zero, logical_one, logical_two], [logical_zero, logical_one, logical_two])
        
        for error in error_detection:
            C_error = expectation_value_tensor(error, logical_zero, logical_zero)
            for tuples in conditions_matrix:
                (first_logical, second_logical) = tuples
                if first_logical == second_logical:
                    if abs(C_error - expectation_value_tensor(error, first_logical, second_logical)) > 1e-5:
                        print('The expectation value of ') 
                        print(str(error)) 
                        print('and' + str(first_logical) + ' is not equal to ' + str(C_error) +  ', the expectation value of the error with Logical 0. It is ') 
                        print(str(expectation_value_tensor(error, first_logical, second_logical)) )
                        return False
                elif first_logical != second_logical:
                    if abs(expectation_value_tensor(error, first_logical, second_logical)) > 1e-16:
                        p = expectation_value_tensor(error, first_logical, second_logical)
                        print('The expectation value of ' + str(error) + ', ' + str(first_logical) + ', and ' + str(second_logical) + ' is not equal to 0. It equals ' + str(p))
                        return False
        return error_detecting
                    
    def is_error_correcting(self):
        error_correcting = True
        error_correction = self.get_error_correction()
        (logical_zero, logical_one, logical_two) = (self.get_logical_zero(), self.get_logical_one(), self.get_logical_two())
        conditions_matrix = generate_square([logical_zero, logical_one, logical_two], [logical_zero, logical_one, logical_two])       
        for error in error_correction:
            C_error = expectation_value_tensor(error, logical_zero, logical_zero)
            for tuples in conditions_matrix:
                (first_logical, second_logical) = tuples
                if first_logical == second_logical:
                    if abs(C_error - expectation_value_tensor(error, first_logical, second_logical)) > 1e-6:
                        error_correcting = False
                elif first_logical != second_logical:
                    if 0 != expectation_value_tensor(error, first_logical, second_logical):
                        error_correcting = False
        return error_correcting
            
            
             
            

###############################################
# Test Operators
###############################################

X = Operator([[0,0,1], [1,0,0], [0,1,0]])


lambda_1 = Operator([[0,1,0], [1,0,0], [0,0,0]])
lambda_2 = Operator([[0,0,1], [0,0,0], [1,0,0]])
lambda_3 = Operator([[0,0,0], [0,0,1], [0,1,0]])
lambda_4 = Operator([[0,complex(0,-1),0], [complex(0,1),0,0], [0,0,0]])
lambda_5 = Operator([[0,0,complex(0,-1)], [0,0,0], [complex(1,0),0,0]])
lambda_6 = Operator([[0,0,0], [0,0,complex(0, -1)], [0,complex(0,1),0]])

vect_objects = [Sym_Vect(4,0,0), Sym_Vect(1,3,0), Sym_Vect(1,0,3), Sym_Vect(0,2,2), Sym_Vect(2,1,1)]

zero = Logical_Bit(vect_objects)

one = applyOperator_logical_bit(X, zero)    

I = Operator([[1,0,0],[0,1,0], [0,0,1]])    
    
    
H_1 = Operator([[1,0,0], [0, -1, 0] , [0,0,0]])
H_2 = Operator([[0,0,0], [0,1,0], [0,0,-1]])

    
a_1 = math.sqrt(10214875/168)

a_0 = math.sqrt(670371601625)

c = math.sqrt(8586854127423000)


list_of_lists = [[35,1,1, a_0/c], [11,1,25, a_1/c], [11, 25,1, a_1/c], [9, 14, 14, 1/c]]
    
logical_zero = generate_logical_bit(list_of_lists)

error_space = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, H_1, H_2]


code_space = Codespace(logical_zero, error_space)

new_logical_zero = Logical_Bit([Sym_Vect(4,0,0, 2), Sym_Vect(0,2,2,2)])

new_logical_one = Logical_Bit([Sym_Vect(0,4,0), Sym_Vect(2,0,2)])





    
    


























        