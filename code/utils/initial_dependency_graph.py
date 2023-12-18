# -*- coding: utf-8 -*-
def dependency_adj_matrix(token_range, postag, length, max_len, matrix):
    adj_postag = []
    for i in range(1, length + 1):
        target_range = token_range[i - 1]
        tag = postag[target_range[1]]
        adj_postag.append(tag)
    for i in range(0, len(adj_postag)):
        if adj_postag[i] != 0:
            for l in range(0, len(adj_postag)):
                if matrix[i][l] == 1:
                    matrix[i][l] = matrix[i][l] + 1
                    if i != l:
                        matrix[l][i] = matrix[l][i] + 1
    p = []
    for i in range(max_len):
        p.append(float(0))
    matrix.insert(0, p)
    matrix.pop()
    for i in range(1, len(matrix)):
        matrix[i].insert(0, float(0))
        matrix[i].pop()
    return matrix


def initial(sentence, token_range, pred_tags, max_len, adj):
    text = sentence.split()
    lengths = len(text)
    adj = dependency_adj_matrix(token_range, pred_tags, lengths, max_len, adj)
    return adj
