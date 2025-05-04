import re
import random
import ast
import operator

def extract_solution(solution_str):
    # TODO: format rewards
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)

    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer

def validate_path(predicted_path, graph_edges):
    """
    Will return true only if you predict a valid path in the graph
    """
    try:
        indices_in_path = [int(n) for n in re.findall(r'\d+', predicted_path)]
        for node_position in range(1, len(indices_in_path)):
            node_u = indices_in_path[node_position - 1]
            node_v = indices_in_path[node_position]
            if f'{node_u},{node_v}' not in graph_edges:
                return False
        return True
    except:
        return False

def compute_exact_match_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for the tree search task
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing the graph, source, destination and, correct path from source to destination
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    ground_truth_path = ground_truth['path']
    graph_edges = ground_truth['graph'].split('|')
    predicted_path = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 4) == 1

    
    if predicted_path is None:
        if do_print:
            print('VAL No path found')
        return 0

    try:
        predicted_path = predicted_path.rstrip().lstrip()
        if predicted_path == ground_truth_path:
            if do_print:
                print(f'VAL Correct path {predicted_path}')
            return score
        else:
            if do_print:
                print(f'VAL Incorrect path: {predicted_path} || Correct path is: {ground_truth_path} || Graph Edges: {graph_edges}')
            return 0
    except:
        if do_print:
            print('VAL Error occurred')
        return 0

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for the tree search task
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing the graph, source, destination and, correct path from source to destination
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    ground_truth_path = ground_truth['path']
    graph_edges = ground_truth['graph'].split('|')
    predicted_path = extract_solution(solution_str=solution_str)

    do_print = random.randint(1, 4) == 1

    # if do_print:
    #     print(f"--------------------------------")
    #     print(f"Graph edges: {graph_edges}")
    #     print(f"Ground truth path: {ground_truth_path}")

    #     print()
    #     print("Model Output:")
    #     print(solution_str)
    #     print()

    #     print(f"Extracted path: {predicted_path}")
    
    if predicted_path is None:
        if do_print:
            print('No path found')
        return 0

    if not validate_path(predicted_path, graph_edges):
        if do_print:
            print('Invalid Path')
        return format_score
    
    try:
        predicted_path = predicted_path.rstrip().lstrip()
        if predicted_path == ground_truth_path:
            if do_print:
                print(f'Correct path {predicted_path}')
            return score
        else:
            if do_print:
                print(f'Incorrect path: {predicted_path} || Correct path is: {ground_truth_path} || Graph Edges: {graph_edges}')
            return format_score
    except:
        if do_print:
            print('Error occurred')
        return format_score