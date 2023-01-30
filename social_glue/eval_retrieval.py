def get_score(data: dict, Y, tops=[1]):
    """get top 1-2-3 scores of data prediction. Need to fix Y in function to compare input data to a Y golden standar.

    Args:
        data (dict): data dictionnary with ranked contents for each query: {query_id:[content_id1, content_id2,...]}
        Y (dict): data dictionnary with GT: {query_id:content_id}


    Returns:
        list: score list [top1, top2, top3] -> number of y in top1 - 2 - 3
    """
    if isinstance(tops,int):
        tops = [tops]
    i = 0
    tops = dict([(top,0) for top in tops])
    for k,ranked_contents in data.items():
        assert k in Y
        gt = Y[k]
        for top in tops:
            if gt in ranked_contents[:top]:
                tops[top] += 1
    tops = dict([(top,t/len(Y)) for (top,t) in tops.items()])
    return tops


if __name__ == "__main__":
    Y = {1:45,2:32,3:12}
    ranked_contents = {1:[0,45],
    2:[0,0,32],
    3:[0,0,0,0,0,0,0,12]
    }
    print(get_score(ranked_contents,Y,tops=[1,2,5,10]))
