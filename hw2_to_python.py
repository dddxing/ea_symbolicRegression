#!/usr/bin/env python
# coding: utf-8

# In[611]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import math
import tqdm
from datetime import datetime

today = datetime.now()
d = today.strftime("%Y-%m-%d-%H-%M")
path = "/home/dxing/Desktop/ea_symbolicRegression/"


filename = "data"
df = pd.read_csv(f'{filename}.txt', header = None, sep=", ", names=["x", 'y'], engine='python')
operators1 = ['+', '-', '*', '/', 'sin', 'cos', 'const', 'x']
operators2 = ['const', 'x']

num_evaluation = 50000
depth = 8

x_s = df["x"].to_numpy()
y_s = df["y"].to_numpy()
y_s = [float(y) for y in y_s]


# In[614]:


def find_all_children(arr):
    """
    random pick a parent node and return it in a list with ALL its children (children, grand-children)  
    """
    
    res = []
    ptr = 0
    n = random.randint(2*len(arr)//5, len(arr)//2) # len(arr)-1
    res.append(n)
    
    while res[ptr] <= len(arr)-1:
        if 2 * res[ptr] <= len(arr)-1:
            res.append(2*res[ptr])
        else: break
        if 2 * res[ptr] + 1 <= len(arr)-1:
            res.append(2*res[ptr]+1)
        else: break
            
        ptr += 1
    return res


def merge(dict1, dict2):
    """merge two dictionaries"""
    res = {**dict1, **dict2}
    return res


def random_generate(depth):
    """
    generate a sequence of binary heap
    INPUTS:
        depth: depth of the binary tree -> int
    
    OUTPUT:
        binary heap
    """
    
    max_length = 2 ** depth - 1
    
    length = random.randint(1, max_length)
    first_half = length//2
    second_half = length - first_half
    
    res = ['nah']
    
    for i in range(1, first_half):
        pick = random.choice(operators1)
        
        if (pick == res[i-1]) and (res[i//2] == '-'):
            tmp = copy.deepcopy(operators1)
            tmp.remove(res[i-1])
            pick = random.choice(tmp)
            
        if pick == "const":
            pick = round(random.random() * 10 + 0.1, 4)
            
        res.append(pick)
        
    for i in range(first_half, length):
        
        pick = random.choices(population=operators2, weights=(0.5, 0.5))[0]
        
        if (pick == res[i-1]) and (res[i//2] == '-'):
            tmp = copy.deepcopy(operators2)
            tmp.remove(res[i-1])
            pick = random.choice(tmp)
            
        if pick == "const":
            pick = round(random.random() * 10 + 0.1, 4)
            
        res.append(pick)
        
    return res


# In[616]:


def mutate(binary_heap, mutate_rate):
    
    binaryheap = copy.deepcopy(binary_heap)
    
    ele = int(len(binaryheap) * mutate_rate)
    
    for i in range(ele):
        ptr = random.randint(1, len(binaryheap)-1)

        if ptr < len(binaryheap)//2:
            pick = random.choice(operators1)

            if (pick == binaryheap[ptr-1]) and (binaryheap[ptr//2] == '-'):
                tmp = copy.deepcopy(operators1)
                tmp.remove(binaryheap[ptr-1])
                pick = random.choice(tmp)       

        else:
            pick = random.choice(operators2)

            if (pick == binaryheap[ptr-1]) and (binaryheap[ptr//2] == '-'):
                tmp = copy.deepcopy(operators2)
                tmp.remove(binaryheap[ptr-1])
                pick = random.choice(tmp)

        if pick == "const":
            pick = round(random.random() * 10 + 0.1, 4)
        
        binaryheap[ptr] = pick

    return binaryheap


def evaluate_binary_heap(binary_heap, x):
    
    """
    Evaluate binary from back to front
    
    """
    
    bh = copy.deepcopy(binary_heap)
    for i in range(len(bh)-1, 0, -1):
        try:
            if bh[i] == 'x':
                bh[i] = x

            elif bh[i] == '+':

                bh[i] = bh[2*i] + bh[2*i+1]

            elif bh[i] == '-':

                bh[i] = bh[2*i] - bh[2*i+1]

            elif bh[i] == '*':

                bh[i] = bh[2*i] * bh[2*i+1]

            elif bh[i] == '/':

                bh[i] = bh[2*i] / bh[2*i+1]

            elif bh[i] == 'sin':

                bh[i] = np.sin(bh[2*i])

            elif bh[i] == 'cos':

                bh[i] = np.cos(bh[2*i])
        except:
#             return random.random()
            return 10
#             pass
#             print(f"Math invalid")
#             print(binary_heap)
            
    return bh[1]


# In[623]:


def calculate_y(x_s, equation):
    calculated_y = []
    for ele in x_s:
        res = evaluate_binary_heap(equation, x=ele)
        calculated_y.append(res)
    return calculated_y



def calculate_mse(y, y_hat):
    """
    use np to calculate mse
    """
#     calculated_y = np.array(calculate_y(x_s, equation=equation))
#     calculated_y = [x.astype(float) for x in calculated_y] # cast to np.float
#     y = [n.astype(float) for n in y]
#     y_hat = [n.astype(float) for n in y_hat]
#     print(len(y) == len(y_hat))
    y = np.array(y)
    y_hat = np.array(y_hat)
    mse = np.square(np.subtract(y, y_hat)).mean()
    return mse  


# In[503]:


def calc_mse(y, y_hat):
    """
    home-made mse calculation
    """
    errors_sq = []
    tmp = 0
    for i in range(len(y)):
        try:
            e = (y[i] - y_hat[i]) ** 2
            tmp = e
        except TypeError:
            pass
        errors_sq.append(tmp)
    return round(sum(errors_sq)/len(errors_sq), 8)





def generate_population(population, depth):
    """
    generate some number of population
    """
    pool = {}
    
    for i in range(population):
        equation = random_generate(depth)
        y_calculated = calculate_y(x_s, equation=equation)
        mse = calc_mse(y_calculated, y_s)
        
        pool[mse] = equation
    
    return pool


# In[564]:


def crossover(parent1, parent2):
    """
    crossover operation
    1. random pick a point at parent1
    2. random pick a point at parent2
    3. completely swap the points and their children
    4. return two trees
    """
    
    p1 = copy.deepcopy(parent1)
    p2 = copy.deepcopy(parent2)
    
    rm_idxs1 = find_all_children(p1) # top node and its grand-grandchildren - random
    rm_idxs2 = find_all_children(p2) # top node and its grand-grandchildren - random
    
#     print(rm_idxs1)
#     print(rm_idxs2)
    
    removed1 = [p1[x] for x in rm_idxs1] 
    removed2 = [p2[x] for x in rm_idxs2]
    
    for idx in rm_idxs1:
        p1[idx] = 'nah'
    
    for idx in rm_idxs2:
        p2[idx] = 'nah' 
        
        
    # if they are in the same length
    if len(removed1) == len(removed2):
        for ele in removed1:
            for i in range(1, len(p2)):
                if p2[i] == "nah":
                    p2[i] = ele
                    break
                    
        for ele in removed2:
            for i in range(1, len(p1)):
                if p1[i] == "nah":
                    p1[i] = ele
                    break
    
    # if not
    elif len(removed1) > len(removed2):

        ptr_rm = 0
        ptr_idx = 0
        while ptr_rm < len(removed1):
            while ptr_idx < len(rm_idxs2): 
                p2[rm_idxs2[ptr_idx]] = removed1[ptr_rm]
                ptr_idx += 1
                ptr_rm += 1
                
            p2.append(removed1[ptr_rm])
            ptr_rm += 1
            
        for ele in removed2:
            for idx in rm_idxs1:
                p1[idx] = ele
                break                   
    else:
        ptr_rm = 0
        ptr_idx = 0
        while ptr_rm < len(removed2):
            while ptr_idx < len(rm_idxs1): 
                p1[rm_idxs1[ptr_idx]] = removed2[ptr_rm]
                ptr_idx += 1
                ptr_rm += 1

            p1.append(removed2[ptr_rm])
            ptr_rm += 1
            
        for ele in removed1:
            for idx in rm_idxs2:
                p2[idx] = ele
                break        

         
    for i in range(1, len(p1)):
        if p1[i] == 'nah':
#             p1[i] = 'x'
            p1[i] = round(random.random() * 10 + 0.1, 4) # not the best
    for i in range(1, len(p2)):
        if p2[i] == 'nah':
#             p2[i] = 'x'
            p2[i] = round(random.random() * 10 + 0.1, 4) # not the best
                
    child1 = p1
    child2 = p2
        
    return [child1, child2]


def mutate_pool(pool: dict, mutate_rate: float)-> dict:
    """
    mutate a pool of population by given mutation rate
    """
    copied_pool = copy.deepcopy(pool)
    
    new_pool = {}
    
    number = int(len(copied_pool) * mutate_rate)
    
    for i in range(number):
        keys = list(copied_pool.keys())
        random.shuffle(keys)
        mutant = mutate(copied_pool[keys[0]], mutate_rate=1)
        y_hat = calculate_y(x_s, equation=mutant)
        mse = calc_mse(y_s, y_hat)
        new_pool[mse] = mutant
    pool = merge(new_pool, copied_pool)
    return pool




def random_search(num, x_s, y_s, depth):
    
    """
    random search
    """
    evaluation = []
    error = []
    final_equation = []
    y_calculated = []
    
    for i in range(num):
        
        if i % 100 == 0:
            print(f"{i/num_evaluation * 100} % complete")

        calculated_y = []
        
        equation = random_generate(depth=depth)
        
        for ele in x_s:
            res = evaluate_binary_heap(equation, x=ele)
            calculated_y.append(res)
            
        calculated_y = np.array(calculated_y)
        try:
            mse = (np.square(y_s - calculated_y)).mean(axis=0)
        except:
            print("raise MSE error")
        
        if len(error) == 0:
            error.append(mse)
            evaluation.append(i)
            final_equation = equation
            y_calculated = calculated_y
            
        elif mse < error[-1]:
            error.append(mse)
            evaluation.append(i)
            final_equation = equation
            y_calculated = calculated_y
            
    return [evaluation, error, final_equation, y_calculated]


# In[589]:


def random_mutation_hill_climber(num, x_s, y_s, depth):
    """
    random hill climber
    """
    
    evaluation = []
    error = []
    final_equation = []
    y_calculated = []
    
    
    first_gen = random_generate(depth=depth)
    y_cal = calculate_y(x_s=x_s, equation=first_gen)
    mse = calculate_mse(y_s, y_cal)
    
    for i in range(0, num):
        
        mutant = mutate(first_gen, mutate_rate=0.01)
        calculated_y = calculate_y(x_s, equation=mutant)
        mse = calculate_mse(calculated_y, y_s)
        
        if len(error) == 0:
            error.append(mse)
            evaluation.append(i)
            final_equation = mutant
            y_calculated = calculated_y

        elif mse < error[-1]:
            error.append(mse)
            evaluation.append(i)
            final_equation = mutant
            y_calculated = calculated_y
            
        if i % 100 == 0:
            print(f"{i/num_evaluation * 100} % complete")
    
    return [evaluation, error, final_equation, y_calculated]


# In[590]:


def random_restart_hill_climber(num_eval, x_s, y_s, depth, num_tries):
    evaluations = []
    errors = []
    final_equation = []
    y_calculated = []
    counter = num_eval
    
    while counter > 0:
        
        first_gen = random_generate(depth=depth)
        y_cal = calculate_y(x_s=x_s, equation=first_gen)
        mse = calculate_mse(y_s, y_cal)
        
        if (num_eval - counter) % 100 == 0:
            print(f"{(num_eval - counter)/num_evaluation*100} % complete")
        
        counter -= 1
    
        if len(errors) == 0:
            errors.append(mse)
            evaluations.append(num_eval - counter)
            final_equation = first_gen
            y_calculated = calculate_y(x_s, equation=first_gen)
        
        while num_tries > 0 and counter > 0:
            
            mutant = mutate(first_gen, mutate_rate=0.01)
            calculated_y = calculate_y(x_s, equation=mutant)
            new_mse = calculate_mse(calculated_y, y_s)
            
            if (num_eval - counter) % 100 == 0:
                print(num_eval - counter)
            
            if new_mse < errors[-1]:
                errors.append(new_mse)
                evaluations.append(num_eval - counter)
                final_equation = mutant
                y_calculated = calculate_y(x_s, mutant)
            else:
                num_tries -= 1
        
        if mse < errors[-1]:
            errors.append(mse)
            evaluations.append(num_eval - counter)
            final_equation = first_gen
            y_calculated = calculate_y(x_s, equation=first_gen)
            
    return [evaluations, errors, final_equation, y_calculated]


# In[548]:


def evol_algo(num_eval, x_s, y_s, depth, init_pop):
    """
    selection 50%
    mutation are built-in to crossover
    """
    
    evaluations = []
    errors = []
    final_equation = []
    y_calculated = []
    
    counter = num_eval

    pool = generate_population(depth=depth, population=init_pop)
    # print(f"pool = {len(pool)}")
    num_keys = len(pool)
    try:
        while counter > 0 and num_keys > 0:
            children_pool = {}

            keys = sorted(pool.keys(), reverse=False) 
#             keys = list(pool.keys())
            for i in range(0, len(keys)-1, 2):
                kid1, kid2 = crossover(pool[keys[i]], pool[keys[i+1]])

                y_hat_kid1 = calculate_y(x_s=x_s, equation=kid1)
                y_hat_kid2 = calculate_y(x_s=x_s, equation=kid2)
    #             children_pool[calculate_mse(y_s, y_hat_kid1)] = kid1
    #             children_pool[calculate_mse(y_s, y_hat_kid2)] = kid2

                children_pool[calc_mse(y=y_s, y_hat=y_hat_kid1)] = kid1
                children_pool[calc_mse(y=y_s, y_hat=y_hat_kid2)] = kid2

            # print(f"children pool = {len(children_pool)}")

            merge_pool = merge(children_pool, pool)
            # print(f"merged pool = {len(merge_pool)}")

            pool = merge_pool

            temp_pool = {}

            keys = sorted(pool.keys(), reverse=False)
            half_keys = keys[0:9*(len(keys)//10)] # selection X%
            num_keys = len(half_keys)
            print(f"# of keys = {num_keys}")

            for i in range(num_keys): 
                temp_pool[keys[i]] = pool[keys[i]]
            # print(f"top pool = {len(temp_pool)}")

            lowest_error = min(half_keys)
            print(f"lowest error = {lowest_error}")
            if (len(errors) == 0) or (lowest_error < errors[-1]):
                evaluations.append(num_eval - counter)
                final_equation = temp_pool[lowest_error]
                y_calculated = calculate_y(x_s, final_equation)
                errors.append(lowest_error)

                with open(f"{path}tmp/ea_curve.txt",'a') as e:
                    e.write(str(num_eval-counter))
                    e.write(', ')
                    e.write(str(lowest_error))
                    e.write('\n')

                with open(f"{path}tmp/ea_yhat.txt",'w') as e:
                    for ele in y_calculated:
                        e.write(str(ele))
                        e.write('\n')

                with open(f"{path}tmp/ea_equation.txt",'w') as e:
                    for ele in final_equation:
                        e.write(str(ele))
                        e.write('\n') 


            pool = temp_pool
            counter -= 1

    #         if counter % 10 == 0:
            print(f"counter = {num_eval - counter}")
    except ValueError:
        pass
    return [evaluations, errors, final_equation, y_calculated]


# In[569]:


def evol_algo_div(num_eval, x_s, y_s, depth, init_pop):
    """
    selection 50%
    mutation are built-in to crossover
    adding some random diversity
    """
    
    evaluations = []
    errors = []
    final_equation = []
    y_calculated = []
    
    counter = num_eval

    pool = generate_population(depth=depth, population=init_pop)
    # print(f"pool = {len(pool)}")
    num_keys = len(pool)
    try:
        while counter > 0 and num_keys > 0:
            children_pool = {}

            keys = sorted(pool.keys(), reverse=False) 
#             keys = list(pool.keys())
            for i in range(0, len(keys)-1, 2):
                kid1, kid2 = crossover(pool[keys[i]], pool[keys[i+1]])

                y_hat_kid1 = calculate_y(x_s=x_s, equation=kid1)
                y_hat_kid2 = calculate_y(x_s=x_s, equation=kid2)

                children_pool[calc_mse(y=y_s, y_hat=y_hat_kid1)] = kid1
                children_pool[calc_mse(y=y_s, y_hat=y_hat_kid2)] = kid2

            # print(f"children pool = {len(children_pool)}")
            
            merge_pool = merge(children_pool, pool)
            # print(f"merged pool = {len(merge_pool)}")
            
            pool = merge_pool

            temp_pool = {}

            keys = sorted(pool.keys(), reverse=False)
            half_keys = keys[0:5*(len(keys)//10)] # selection X%
            num_keys = len(half_keys)
            if num_keys < init_pop:
                new_pool = generate_population(depth=depth, population=(init_pop - num_keys)//2)
                temp_pool = merge(temp_pool, new_pool)
            print(f"# of keys = {num_keys}")

            for i in range(num_keys): 
                temp_pool[keys[i]] = pool[keys[i]]
            print(f"top pool = {len(temp_pool)}")

            lowest_error = min(half_keys)
            print(f"lowest error = {lowest_error}")
            if (len(errors) == 0) or (lowest_error < errors[-1]):
                evaluations.append(num_eval - counter)
                final_equation = temp_pool[lowest_error]
                y_calculated = calculate_y(x_s, final_equation)
                errors.append(lowest_error)

                with open(f"{path}tmp/ea_d_curve.txt",'a') as e:
                    e.write(str(num_eval-counter))
                    e.write(', ')
                    e.write(str(lowest_error))
                    e.write('\n')

                with open(f"{path}tmp/ea_d_yhat.txt",'w') as e:
                    for ele in y_calculated:
                        e.write(str(ele))
                        e.write('\n')

                with open(f"{path}tmp/ea_d_equation.txt",'w') as e:
                    for ele in final_equation:
                        e.write(str(ele))
                        e.write('\n')

            pool = temp_pool
            counter -= 1

    #         if counter % 10 == 0:
            print(f"counter = {num_eval - counter}")
    except ValueError:
        pass
    return [evaluations, errors, final_equation, y_calculated]


# In[709]:


def evol_algo_div_mut(num_eval, x_s, y_s, depth, init_pop, mutate_rate):
    """
    selection 50%
    mutation are built-in to crossover
    adding some random diversity
    adding mutation in the pool
    """
    
    evaluations = []
    errors = []
    final_equation = []
    y_calculated = []
    
    counter = num_eval

    pool = generate_population(depth=depth, population=init_pop)
    # print(f"pool = {len(pool)}")
    num_keys = len(pool)
    try:
        while counter > 0 and num_keys > 0:
            children_pool = {}

            keys = sorted(pool.keys(), reverse=False) 
#             keys = list(pool.keys())
            for i in range(0, len(keys)-1, 2):
                kid1, kid2 = crossover(pool[keys[i]], pool[keys[i+1]])

                y_hat_kid1 = calculate_y(x_s=x_s, equation=kid1)
                y_hat_kid2 = calculate_y(x_s=x_s, equation=kid2)

                children_pool[calc_mse(y=y_s, y_hat=y_hat_kid1)] = kid1
                children_pool[calc_mse(y=y_s, y_hat=y_hat_kid2)] = kid2

            # print(f"children pool = {len(children_pool)}")
            
            merge_pool = merge(children_pool, pool)
            
            mutated_pool = mutate_pool(pool=merge_pool, mutate_rate=mutate_rate)
#             print(f"merged pool = {len(merge_pool)}")
            
            pool = mutated_pool

            temp_pool = {}

            keys = sorted(pool.keys(), reverse=False)
            half_keys = keys[0:5*(len(keys)//10)] # selection X%
            num_keys = len(half_keys)

            if num_keys < init_pop:
                new_pool = generate_population(depth=depth, population=(init_pop - num_keys)//2)
                temp_pool = merge(temp_pool, new_pool)
            print(f"# of keys = {num_keys}")

            for i in range(num_keys): 
                temp_pool[keys[i]] = pool[keys[i]]
            # print(f"top pool = {len(temp_pool)}")

            lowest_error = min(half_keys)
            print(f"lowest error = {lowest_error}")
            if (len(errors) == 0) or (lowest_error < errors[-1]):

                evaluations.append(num_eval - counter)
                final_equation = temp_pool[lowest_error]
                y_calculated = calculate_y(x_s, final_equation)

                plt.figure(figsize=(10,10))
                plt.gca().set_aspect('equal')
                plt.plot(x_s, y_s, 'r--',label="Given Data")
                plt.plot(x_s,y_calculated, label= "Learning Data")
                plt.title(f"MSE = {round(lowest_error, 2)}")
                plt.savefig(f'{path}screenshots/foo_{num_eval - counter}.png', 
                        bbox_inches='tight')
                plt.close()
                path = "/home/dxing/Desktop/ea_symbolicRegression/"
                
                with open(f"{path}tmp/ea_dm_curve.txt",'a') as e:
                    e.write(str(num_eval-counter))
                    e.write(', ')
                    e.write(str(lowest_error))
                    e.write('\n')

                with open(f"{path}tmp/ea_dm_yhat.txt",'w') as e:
                    for ele in y_calculated:
                        e.write(str(ele))
                        e.write('\n')

                with open(f"{path}tmp/ea_dm_equation.txt",'w') as e:
                    for ele in final_equation:
                        e.write(str(ele))
                        e.write('\n')


                errors.append(lowest_error)

            pool = temp_pool
            counter -= 1

    #         if counter % 10 == 0:
            print(f"counter = {num_eval - counter}")
    except ValueError:
        pass
    return [evaluations, errors, final_equation, y_calculated]


# In[710]:


res_ea_div_mut = evol_algo_div_mut(num_eval=num_evaluation, x_s=x_s, y_s=y_s, depth=depth, init_pop=20, mutate_rate=0.3)
# res_ea_div = evol_algo_div(num_eval=num_evaluation, x_s=x_s, y_s=y_s, depth=depth, init_pop=20)
# res_ea = evol_algo(num_eval=num_evaluation, x_s=x_s, y_s=y_s, depth=depth, init_pop=20)
# res_rrhc = random_restart_hill_climber(num_eval=num_evaluation, x_s=x_s, y_s=y_s, depth=8, num_tries=10)
# res_rmhc = random_mutation_hill_climber(num=num_evaluation, x_s=x_s, y_s=y_s, depth=8)
# res_rs = random_search(num=num_evaluation, x_s=x_s, y_s=y_s, depth=8)


plt.figure(figsize=(10,10))
# plt.scatter(res_rs[0], res_rs[1], color='purple', label='random search')
# plt.plot(res_rs[0], res_rs[1], color='purple')

# plt.scatter(res_rmhc[0], res_rmhc[1], color='orange', label='random mutation hill climber')
# plt.plot(res_rmhc[0], res_rmhc[1], color='orange')

# plt.scatter(res_rrhc[0], res_rrhc[1], color='blue', label='random restart hill climber')
# plt.plot(res_rrhc[0], res_rrhc[1], color='blue')

# plt.scatter(res_ea[0], res_ea[1], color='green', label='EA')
# plt.plot(res_ea[0], res_ea[1], color='green')

plt.scatter(res_ea_div_mut[0], res_ea_div_mut[1], color='red', label='EA_mut_div')
plt.plot(res_ea_div_mut[0], res_ea_div_mut[1], color='red')

plt.legend()

plt.ylim(0, 100)
plt.xscale('log')
plt.ylabel("Fitness")
plt.xlabel("Evaluation");
plt.show()


plt.figure(figsize=(10,10))
plt.plot(x_s, y_s, label="acutal data");
# plt.plot(x_s, res_rs[3], label="random search");
# plt.plot(x_s, res_rrhc[3], label="random restart HC");
# plt.plot(x_s, res_rmhc[3], label="random mutation HC");
# plt.plot(x_s, res_ea[3], label="EA");
# plt.plot(x_s, res_ea[3], label="EA_div");
plt.plot(x_s, res_ea_div_mut[3], label="EA_div_mut");
plt.legend()
plt.show()



def save_data(result, title):

    path = "/home/dxing/Desktop/ea_symbolicRegression/tmp/" 
    
    df_plot = pd.DataFrame(data={'evaluation': result[0], 'mse': result[1]})
    df_graph = pd.DataFrame(data={"y_cal": result[3]})
    
    with open(f"{path}{d}_{title}_e{num_evaluation}_final_equation.txt",'w') as e:
            for ele in result[2]:
                e.write(str(ele))
                e.write('\n')
                

    df_plot.to_csv(f"{path}{d}_{title}_e{num_evaluation}_plot.csv", index=False)
    df_graph.to_csv(f"{path}{d}_{title}_e{num_evaluation}_graph.csv", index=False)


# In[610]:


# save_data(res_rs,title="rs")
# save_data(res_rmhc,title="rmhc")
# save_data(res_rrhc,title="rrhc")
# save_data(res_ea,title="ea")
# save_data(res_ea_div,title="ea with div")
save_data(res_ea_div_mut,title="ea_div_mut")




