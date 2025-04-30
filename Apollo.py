import gurobipy
from gurobipy import GRB
import argparse
import random
import os
import numpy as np
import torch
from helper import get_a_new2
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
import datetime

parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    '--problem',
    default = "is",
    help='MILP instance type to process.',
)
parser.add_argument(
    '-g', '--gpu',
    help='Number of parallel jobs.',
    type=int,
    default=3,
)
args = parser.parse_args()
TaskName=args.problem
DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

TestNum=100

def test_hyperparam(task, step):
    '''
    set the hyperparams
    k_0, k_1, delta
    '''
    if task=="ip":
        if step == 0:
            return 100,20,50
        elif step == 1:
            return 40,15,50
        elif step == 2:
            return 20,15,30
        elif step == 3:
            return 1,5,10
        else:
            return 10,20,5
   


def get_graph_representation(model):

    A, v_map, v_nodes, c_nodes, b_vars=get_a_new2(model)
    constraint_features = c_nodes.cpu()
    constraint_features[np.isnan(constraint_features)] = 1 #remove nan value
    variable_features = v_nodes
    edge_indices = A._indices()
    edge_features = A._values().unsqueeze(1)
    edge_features=torch.ones(edge_features.shape)
    return A, v_map, v_nodes, c_nodes, b_vars, \
        constraint_features, variable_features, edge_indices, edge_features


def variable_alignment(v_map, b_vars, BD, fixing_status):
   
    all_varname=[]
    for name in v_map:
        all_varname.append(name)
        fixing_status[name] = -1
    binary_name=[all_varname[i] for i in b_vars]
    scores=[] # get a list of (index, VariableName, Prob, -1, type)
    for i in range(len(v_map)):
        type="C"
        if all_varname[i] in binary_name:
            type='BINARY'
            fixing_status[all_varname[i]] = 2
        scores.append([i, all_varname[i], BD[i].item(), -1, type])
    
    scores.sort(key=lambda x:x[2],reverse=True)
    scores=[x for x in scores if x[4]=='BINARY'] # get binary
   
    return scores, fixing_status


def fix_variable(scores, fixing_status, k_0, k_1, delta):
   
    count1=0
    for i in range(len(scores)):
        if count1<k_1:
            scores[i][3] = 1
            count1+=1
            fixing_status[scores[i][1]] = 1

    scores.sort(key=lambda x: x[2], reverse=False)
    count0 = 0
    for i in range(len(scores)):
        if count0 < k_0:
            scores[i][3] = 0
            count0 += 1
            fixing_status[scores[i][1]] = 1

    print(f'instance: {test_ins_name}, '
          f'fix {k_0} 0s and '
          f'fix {k_1} 1s, delta {delta}. ')
   
    return scores, fixing_status


def prediction_correction():
    #set log folder
    time = datetime.datetime.now()
    solver='GRB'
    test_task = f'{TaskName}_{solver}_Predect&Search'
    if not os.path.isdir(f'./logs'):
        os.mkdir(f'./logs')
    if not os.path.isdir(f'./logs/{TaskName}'):
        os.mkdir(f'./logs/{TaskName}')
    if not os.path.isdir(f'./logs/{TaskName}/{test_task}_{time}'):
        os.mkdir(f'./logs/{TaskName}/{test_task}_{time}')
    log_folder=f'./logs/{TaskName}/{test_task}_{time}'

   
    from GCN import GNNPolicy
    # model_name=f'{TaskName}.pth'
    pathstr = f'pretrain/{args.problem}_train/model_best.pth'
    policy = GNNPolicy().to(DEVICE)
    state = torch.load(pathstr, map_location=torch.device('cuda:0'))
    policy.load_state_dict(state)

    log_file = open(f'{log_folder}/test.log', 'wb')

    obj_list = []
    time_list = []
    node_list = []
    time_limit = [100, 100, 200, 600]
    best_obj_list = []
    import time 

    sample_names = sorted(os.listdir(f'./instance/test/{TaskName}'))
    for ins_num in range(min(len(sample_names),TestNum)):
        test_ins_name = sample_names[ins_num]
        ins_name_read = f'./instance/test/{TaskName}/{test_ins_name}'
        os.makedirs(f'{log_folder}/{test_ins_name}', exist_ok=True)
        if args.problem == 'is' or args.problem == 'ca':
            best_obj = 0
        else:
            best_obj = 1000000
        
        t = time.time()
        fixing_status = {} # 2 unfix; 0 fixed; 1 prefix with inequality; -1 not need to fix

        for step in range(len(time_limit)):
            A, v_map, v_nodes, c_nodes, b_vars, \
                constraint_features, variable_features, edge_indices, edge_features = get_graph_representation(ins_name_read)

            #! prediction
            BD = policy(
                constraint_features.to(DEVICE),
                edge_indices.to(DEVICE),
                edge_features.to(DEVICE),
                variable_features.to(DEVICE),
            ).sigmoid().cpu().squeeze()

            k_0,k_1,delta = test_hyperparam(TaskName, step)
           
            scores, fixing_status = variable_alignment(v_map, b_vars, BD, fixing_status)

            scores, fixing_status = fix_variable(scores, fixing_status, k_0,k_1,delta)

            #! read instance
            gurobipy.setParam('LogToConsole', 1)  # hideout
            m = gurobipy.read(ins_name_read)
            m_ps = m.copy()
        
            m_ps.Params.TimeLimit = time_limit[step]
            m_ps.Params.Threads = 1
            m_ps.Params.MIPFocus = 1
            m_ps.Params.LogFile = f'{log_folder}/{test_ins_name}/{step}.log'

            #! trust region method implemented by adding constraints
            instance_variabels = m_ps.getVars()
            instance_variabels.sort(key=lambda v: v.VarName)
            variabels_map = {}
            
            for v in instance_variabels:  # get a dict (variable map), varname:var clasee
                variabels_map[v.VarName] = v
            
            alphas = []

            for i in range(len(scores)):
                tar_var = variabels_map[scores[i][1]]  # target variable <-- variable map
                x_star = scores[i][3]  # 1,0,-1, decide whether need to fix
                if x_star < 0 or fixing_status[scores[i][1]] != 2:
                    continue
                  
                tmp_var = m_ps.addVar(name=f'alp_{tar_var}', vtype=GRB.CONTINUOUS)
                alphas.append(tmp_var)
                m_ps.addConstr(tmp_var >= tar_var - x_star, name=f'alpha_up_{i}')
                m_ps.addConstr(tmp_var >= x_star - tar_var, name=f'alpha_dowm_{i}')

                
            all_tmp = 0
            for tmp in alphas:
                all_tmp += tmp
            m_ps.addConstr(all_tmp <= delta, name="sum_alpha")
            m_ps.optimize()
            t = time.time() - t

            mvars = m.getVars()
         
            scores.sort(key=lambda x: x[2], reverse=False)
            
            counting = 0
            fix_count = 0
            for i in range(len(scores)):
                if scores[i][3] != 0:
                    continue
                
                if fixing_status[scores[i][1]] == 1 and scores[i][3] == 0:
                    var = m.getVarByName(scores[i][1])
                    if scores[i][3] == m_ps.getVarByName(scores[i][1]).x:
                        fixing_status[scores[i][1]] = 0
                        var.lb = m_ps.getVarByName(scores[i][1]).x
                        var.ub = m_ps.getVarByName(scores[i][1]).x
                        fix_count += 1
                    else:
                        fixing_status[scores[i][1]] = 2
                counting += 1

                if counting >= k_0:
                    break
            print(f'inspect {counting} vars and fix {fix_count} to be 0')

            counting = 0
            fix_count = 0
            scores.sort(key=lambda x: x[2], reverse=True)
            for i in range(len(scores)):
                if scores[i][3] != 1:
                    continue
                
                # for the vars prefixed to be 0
                if fixing_status[scores[i][1]] == 1 and scores[i][3] == 1:
                    var = m.getVarByName(scores[i][1])
                    if scores[i][3] == m_ps.getVarByName(scores[i][1]).x:
                        fixing_status[scores[i][1]] = 0
                        var.lb = m_ps.getVarByName(scores[i][1]).x
                        var.ub = m_ps.getVarByName(scores[i][1]).x
                        fix_count += 1
                    else:
                        fixing_status[scores[i][1]] = 2
                counting += 1

                if counting >= k_1:
                    break
            print(f'inspect {counting} vars and fix {fix_count} to be 1')

            m.write(f'{log_folder}/{test_ins_name}/{step}.lp')
            ins_name_read = f'{log_folder}/{test_ins_name}/{step}.lp'
            

            if args.problem == 'is' or args.problem == 'ca':
                if m_ps.ObjVal > best_obj:
                    best_obj = m_ps.ObjVal
            else:
                if m_ps.ObjVal < best_obj:
                    best_obj = m_ps.ObjVal

            st = f' {m_ps.ObjVal} {t} {m_ps.NodeCount}\n'
            log_file.write(st.encode())
            log_file.flush()
            
            obj_list.append(m_ps.ObjVal)
            time_list.append(t)
            node_list.append(m_ps.NodeCount)

        best_obj_list.append(best_obj)



    st = f' {np.mean(obj_list)} {np.mean(time_list)} {np.mean(node_list)} {np.mean(best_obj_list)}\n'
    log_file.write(st.encode())
    log_file.flush()

    st = f' {[test_hyperparam(TaskName, step) for step in range(len(time_limit))]}\n'
    log_file.write(st.encode())
    log_file.flush()








