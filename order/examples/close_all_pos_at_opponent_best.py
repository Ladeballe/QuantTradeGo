# from quant_trade_go.order_handler import OrderHandlerV1
#
#
# if __name__ == '__main__':
#     oh = OrderHandlerV1()
#     oh._get


def solution(s):
    l, r = 0, 0
    max_string = s[l: r]
    max_length = len(s[l: r])
    while r < len(s):
        r += 1
        if len(s[l: r]) > max_length:
            max_length = len(s[l: r])
            max_string = s[l: r]
        while r < len(s) and s[r] in s[l: r]:
            l += 1
        if len(s[l: r]) > max_length:
            max_length = len(s[l: r])
            max_string = s[l: r]
    return max_string, max_length


if __name__ == '__main__':
    print(solution('aabcdcbeaa'))
