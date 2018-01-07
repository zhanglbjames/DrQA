## Reading Notes

1. TD-IDF
> http://www.ruanyifeng.com/blog/2013/03/tf-idf.html

2. N-gram
> http://blog.csdn.net/baimafujinji/article/details/51281816
> https://www.cnblogs.com/zhangkaikai/p/6669750.html
> https://www.ibm.com/developerworks/cn/opensource/os-cn-pythonwith/

3. with expression
> http://blog.csdn.net/suwei19870312/article/details/23258495/

4. init__.py
> http://blog.csdn.net/djangor/article/details/39673659
> http://www.cnpythoner.com/post/2.html

5. functools.partial
> http://www.wklken.me/posts/2013/08/18/python-extra-functools.html

6. multiprocessing package
> https://www.cnblogs.com/vamei/archive/2012/10/12/2721484.html
> http://blog.csdn.net/quqiuzhu/article/details/51156454

7. why multiprocessing has better performance compares to threads
> 为什么Python中多进程要比多线程性能要高呢？
> 这是因为Python有一个全局解释器锁PIL，导致每个Python进程中只能同时最多执行一个线程，因此Python多线程并不能改善性能，
不能发挥多核CPU的优势，所以使用多进程的方式，每个进程之间不受PIL的影响。所以使用内嵌单线程的多进程程序能提高Python程序的性能，
多进程之间使用pipeline和Queue以及共享内存（Array，Value进程安全）来进行通信。

8. pooling api
> https://segmentfault.com/a/1190000008118997

9. Pexpect command line interactive module
> http://blog.csdn.net/sdustliyang/article/details/23373485

10. scipy.sparse.csr_matrix: Compressed Sparse Row matrix
> http://blog.csdn.net/u013010889/article/details/53305595
> http://blog.csdn.net/pipisorry/article/details/41762945
> http://blog.csdn.net/u014471250/article/details/78262926?locationNum=1&fps=1

11. np.array(binary.sum(1)).squeeze()
> http://blog.csdn.net/qq403977698/article/details/47254539

12. np.axis
> http://blog.csdn.net/fangjian1204/article/details/53055219

13. 多维数组的例子 
> https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html#numpy.squeeze

> np.array([[[0], [1], [2]]]) 从里往外看维度的长度为(1,3,1)，即三维，每一维的长度为1,3,1</br>
> ---------|||</br>
> ---------131