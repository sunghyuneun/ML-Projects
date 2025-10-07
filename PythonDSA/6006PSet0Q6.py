def count_long_subarrays(A):
    longest_len = 0
    long_count = 0
    current_len = 0
    for i in range(len(A)):
        if i == 0:
            pass
        else:
            if A[i] > A[i-1]:
                current_len += 1
                if current_len > longest_len:
                    longest_len = current_len
                    long_count = 1
                elif current_len == longest_len:
                    long_count += 1
            else:
                current_len = 1
    return long_count