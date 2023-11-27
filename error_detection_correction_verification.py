# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:31:01 2022

@author: xzavi
"""

import vector_package
import math
import numpy

def int_checker(user_input_string, positive = False):
    actual_input = input(user_input_string)
    marker = True
    while marker:
        try:
            actual_input = float(actual_input)
        except: 
            print("What you entered is not an integer. Please enter an integer value.")
            print('---------------------------------------------------------')
            return int_checker(user_input_string)
        else:
            if float(actual_input) == int(actual_input):
                actual_input = int(actual_input)
                if positive:
                    if actual_input > 0:
                        marker = False
                    else:
                        print("What you entered is a non-positive integer. Please enter a positive integer.")
                        print('---------------------------------------------------------')
                        int_checker(user_input_string, positive)
                marker = False
            else:
                print("What you entered is not an integer. Please enter a positive integer value.")
                print('---------------------------------------------------------')
                return int_checker(user_input_string, positive)
    return actual_input


def float_checker(user_input_string, positive = False):
    actual_input = input(user_input_string)
    marker = True
    while marker:
        try:
            actual_input = float(actual_input)
        except: 
            print("What you entered is not an acceptable number. Please enter a numerical value.")
            print('---------------------------------------------------------')
            return float_checker(user_input_string)
        else:
            if positive:
                if actual_input > 0:
                    marker = False
                else:
                    print("What you entered is a non-positive real number. Please enter a positive real number.")
                    return float_checker(user_input_string)
            else:
                marker = False
    return actual_input

def complex_checker(user_input_string, positive = False):
    actual_input = input(user_input_string)
    marker = True
    while marker:
        try:
            actual_input = complex(actual_input)
        except: 
            print("What you entered is not an acceptable number. Please enter a numerical value.")
            print('---------------------------------------------------------')
            return complex_checker(user_input_string)
        else:
            if positive:
                if actual_input > 0:
                    marker = False
                else:
                    print("What you entered is a non-positive real number. Please enter a positive real number.")
                    return complex_checker(user_input_string)
            else:
                marker = False
    return actual_input




def input_checker(user_input_string, list_of_acceptable_characters):
    actual_input = input(user_input_string)
    if actual_input in list_of_acceptable_characters:
        return actual_input
    else:
        print("Your input is not one of the acceptable inputs. Please re-enter an acceptable value.")
        print('---------------------------------------------------------')
        return input_checker(user_input_string, list_of_acceptable_characters)




marker = True

print("To start, please specify the number of basis vectors in the Logical 0 codeword.")
number_of_basis_vectors = input("How many basis vectors are in the Logical 0 codeword. Must be a positive integer: ")


while marker:
    try:
        number_of_basis_vectors = float(number_of_basis_vectors)
    except: 
        print("What you entered is not an integer. Please enter an integer value.")
        print('---------------------------------------------------------')
        number_of_basis_vectors = input("How many basis vectors are in the Logical 0 codeword. Must be a positive integer: ")
    else:
        if float(number_of_basis_vectors) == int(number_of_basis_vectors):
            number_of_basis_vectors = int(number_of_basis_vectors)
            if number_of_basis_vectors > 0:
                marker = False
            else:
                print("What you entered is a non-positive integer. Please enter a positive integer.")
                print('---------------------------------------------------------')
                number_of_basis_vectors = input("How many basis vectors are in the Logical 0 codeword. Must be a positive integer: ")
        else:
            print("What you entered is not an integer. Please enter a positive integer value.")
            print('---------------------------------------------------------')
            number_of_basis_vectors = input("How many basis vectors are in the Logical 0 codeword. Must be a positive integer: ")
            
        
print('---------------------------------------------------------')
print("You will now construct " + str(number_of_basis_vectors) + " symmetric vectors.")
list_of_basis_vectors = []
for number in range(1, number_of_basis_vectors + 1):
    satisfication = False
    while not satisfication:
        print("For basis vector number " + str(number))
        zero_power = int_checker("The tensor power of 0 is: ")
        one_power = int_checker("The tensor power of 1 is: ")
        two_power = int_checker("The tensor power of 2 is: ")
        coefficient = complex_checker("Coefficient is: ")
        new_sym_vector = vector_package.Sym_Vect(zero_power, one_power, two_power, coefficient)
        print(str(new_sym_vector))
        yes_or_no = input_checker("Is this the vector that you want? Enter Y for yes or N for no: ", ['Y', 'y', 'N', 'n'])
        if yes_or_no == 'Y' or yes_or_no == 'y':
            list_of_basis_vectors.append(new_sym_vector)
            satisfication = True
        elif yes_or_no == 'N' or yes_or_no == 'n':
            print("Please re-enter the information for basis vector number " + str(number))
            
logical_zero = vector_package.Logical_Bit(list_of_basis_vectors)

print("The logical 0 codeword that you generated is given by: ")
print(str(logical_zero))




######################################################################################################
lambda_1 = vector_package.Operator([[0,1,0], [1,0,0], [0,0,0]])
lambda_2 = vector_package.Operator([[0,0,1], [0,0,0], [1,0,0]])
lambda_3 = vector_package.Operator([[0,0,0], [0,0,1], [0,1,0]])
lambda_4 = vector_package.Operator([[0,complex(0,-1),0], [complex(0,1),0,0], [0,0,0]])
lambda_5 = vector_package.Operator([[0,0,complex(0,-1)], [0,0,0], [complex(1,0),0,0]])
lambda_6 = vector_package.Operator([[0,0,0], [0,0,complex(0, -1)], [0,complex(0,1),0]])
H_1 = vector_package.Operator([[1,0,0], [0, -1, 0] , [0,0,0]])
H_2 = vector_package.Operator([[0,0,0], [0,1,0], [0,0,-1]])

error_space = [lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, H_1, H_2]

print('---------------------------------------------------------')

print("The current error space is lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, H_1, H_2.")

#######################################################################################################

code_space = vector_package.Codespace(logical_zero, error_space)

print('---------------------------------------------------------')
print("Now, you can see if the codespace corrects and/or detects the errors in the error space.")

print("To check if the code is error detecting, press 1. To check if the code is error correcting, press 2.")

checking_flag = True

while checking_flag:
    checking_status = input_checker("Input 1 or 2: ", ['1', '2'])
    if checking_status == '1':
        pass_or_fail = code_space.is_error_detecting()
        if pass_or_fail == True:
            print("Congratulations, the codespace is error detecting! ")
    elif checking_status == '2':
        pass_or_fail = code_space.is_error_correcting()
        if pass_or_fail == True:
            print("Congratulations, the codespace is error correcting! ")
            
    next_step = input_checker("Would you like to perform anymore checks? Enter Y for yes and N for no: ", ['Y', 'y', 'N', 'n'])
    if next_step == 'N' or next_step == 'n':
        checking_flag = False
        print("Thank you for using this software!")
        
        

    

















































