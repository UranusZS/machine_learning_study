#!/usr/bin/env python
# coding=utf-8

# ref: http://www.cnblogs.com/hanahimi/p/4692668.html
# 完全树 最小堆
class CompleteTree(list):
    def siftdown(self,i):
        """ 对一颗完全树进行向下调整，传入需要向下调整的节点编号i
        当删除了最小的元素后，当新增加一个数被放置到堆顶时，
        如果此时不符合最小堆的特性，则需要将这个数向下调整，直到找到合适的位置为止"""
        n = len(self)
        # 当 i 节点有儿子（至少是左儿子时），并且有需要调整时，循环执行
        t = 0
        while i*2+1<n:
            # step 1：从当前结点，其左儿子，其右儿子中找到最小的一个，将其编号传给t
            if self[i] > self[i*2+1]: 
                t = i*2+1
            else: t = i
            
            # 如果有右儿子，则再对右儿子进行讨论
            if i*2+2<n:
                if self[t] > self[i*2+2]: t = i*2+2
            
            # step 2：把最小的结点中的元素和结点i的元素交换
            if t != i:
                self[t],self[i] = self[i],self[t]
                i = t   # 更新i为刚才与它交换的儿子结点的编号，以便接下来继续向下调整
            else:
                break   # 说明当前父结点已经比两个子结点要小，结束调整
        
    def siftup(self,i):
        """ 对一棵完全树进行向上调整，传入一个需要向上调整的结点编号i
            当要添加一个新元素后，对堆底（最后一个）元素进行调整 """
        if i==0: return
        n = len(self)
        if i < 0: i += n
        # 注意，由于堆的特性，不需要考虑左儿子结点的情况
        # 由于父结点绝对比子结点小所以只需要比较一次
        while i!=0:
            if self[i]<self[(i-1)/2]:
                self[i],self[(i-1)/2] = self[(i-1)/2],self[i]
            else:
                break
            i = (i-1)/2     # 更新i为其父结点编号，从而便于下一次继续向上调整
    
    def shufflePile(self):
        """ 在当前状态下，对树调整使其成为一个堆 """
        # 从"堆底"往"堆顶"进行向下调整，使得最小的元素不断上升
        # 这样可以使得i结点以下的堆是局部最小堆
        for i in range((len(self)-2)/2,-1,-1):    # n/2,...,0
            self.siftdown(i)

    def deleteMin(self):
        """ 删除最小元素 """
        t = self[0]     # 用一个临时变量记录堆顶点的
        self[0] = self[-1]  # 将堆的最后一个点赋值到堆顶
        self.pop()      # 删除最后一个元素
        self.siftdown(0)    # 向下调整
        return t
    
    
    def heapsort(self):
        """ 对堆中元素进行堆排序操作 """
        n = len(self)
        s = []
        while n>0:
            s.append(self.deleteMin())
            n -= 1
        # 由于堆中的元素已全部弹出，将排序好的元素拼接到原来的堆中
        self.extend(s)  
        
if __name__=="__main__":

    a = [99,5,36,7,22,17,92,12,2,19,25,28,1,46]
    ct = CompleteTree(a)
    print ct

    ct.shufflePile()
    print ct

    s = ct.heapsort()
    print ct
