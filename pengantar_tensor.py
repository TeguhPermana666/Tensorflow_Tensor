"""
tensor adalah=>array multidimensi dengan type seragam(disebut:dtype)
dtypes dapat dilihat dengan tf.dtypes.DType

=>jika terbiasa dengan  numpy , tensor (semacam) seperti np.arrays

"""
import tensorflow as tf
import numpy as np
#tensor skalar/rank-0=>berisi satu nilai dan tidak ada axis
rank_0_tensor=tf.constant(4)
print(rank_0_tensor)
#tensor vektor /rank-1=>berisi daftar nilai yang memiliki satu axis
rank_1_tensor=tf.constant([2.0,3.0,4.0])
print(rank_1_tensor)
#tensor matriks/rank-2=>berisi dua axis
rank_2_tensor=tf.constant([[1,2],[2,3],[3,3]])
print(rank_2_tensor)
#tensor 3 axis
rank_3_tensor= tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])
print(rank_3_tensor)

#convert dengan numpy dengan tensor
print("\nNUMPY:\n")
print(np.array(rank_1_tensor))
print(np.array(rank_2_tensor))
print(np.array(rank_3_tensor))

#matematika
a=tf.constant([
  [1,2],[3,1]
  ])
b=tf.constant([
  [1,1],[2,1]
  ])
#could have also said tf.ones[2,2]

print(tf.add(a, b),"\n")
print(tf.multiply(a, b),"\n")
print(tf.matmul(a, b),"\n")


print(a + b, "\n") # element-wise addition
print(a * b, "\n") # element-wise multiplication
print(a @ b, "\n") # matrix multiplication
"""
tensor terdapat banyak jenis tipe data termasuk juga"
-string
-bilangan kompleks

pada matriks array tensor dllnya memiliki nilai di setiap axisnya yakni sama namun bagimana sekarang
mengaturnya untuk nilai pada setiap axis yang berbeda:
=>ragged tensor
=>sparse tensor 
"""
#Tensor digunakan dalam semua jenis operasi (ops).

c=tf.constant([[1.0,2.0],[10.0,5.0]])
#find the max value
print(tf.reduce_max(c).numpy())
#find the index max value
print(tf.argmax(c).numpy())
#compute the softmax=>digunakan untuk normalisasi eksponensial function dalam multiple dimension
print(tf.nn.softmax(c))
"""
Bentuk : Panjang (jumlah elemen) dari masing-masing sumbu tensor.
Rank : Jumlah sumbu tensor. Sebuah skalar memiliki rank 0, sebuah vektor memiliki rank 1, sebuah matriks memiliki rank 2.
Sumbu atau Dimensi : Dimensi tertentu dari sebuah tensor.
Ukuran : Jumlah total item dalam tensor, vektor bentuk produk.
"""
rank_4_tensor=tf.zeros([3,2,4,5])
#[3,      2,      4,        5]
# batch   lebar   panjang   fitur
print(rank_4_tensor)
print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())#.numpy() digunakan untuk mengambil nilai dari list return value pada function tensor

#pengindeksan
"""
1.pengindeksan di tensor mirip dengan yang ada pada numpy
-indeks dimulai dari 0
-indeks negatif yang berarti di hitung dari n
-index : => irisan (start:stop:step)
"""
print("awal: ",rank_1_tensor[0].numpy())
print("kedua",rank_1_tensor[1].numpy())
print("akhir",rank_1_tensor[-1].numpy())

rank_1_tensor2 = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print("Nilai: ",rank_1_tensor2.numpy())

print("semua:",rank_1_tensor2[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())

# Get row and column tensors
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")

#rank 3
print(rank_3_tensor[:, :, 4])

#Memanipulasi shapes/bentuk
sebelum_reshape=tf.constant([[1],[2],[3]])
print(sebelum_reshape.shape.as_list())#bisa convert ke list
sesudah_reshape=tf.reshape(sebelum_reshape, [1,3])
print(sebelum_reshape)
print(sebelum_reshape.shape)#shape =>mengetahui sebuah bentuk dari tensor

print(sesudah_reshape)
print(sesudah_reshape.shape)

#=>mensama ratakan rank_3_tensor(matriks)
print(tf.reshape(rank_3_tensor, [-1,2]))#data di kalkulasi berdasrkan start:stop:step, yg mana secara default=0 untuk step, start and sto=>:
"""
biasanya tf.reshape()digunakan untuk menggabungkan atau mengurangi axis
 
"""
print(tf.reshape(rank_3_tensor, [3*2,5]))
print(tf.reshape(rank_3_tensor, [3,-1]))
#atau bisa secara langsung di indexing namun diharuskan untuk merunjuk ke harmonisan dan kestabilan 
# Bad examples: don't do this

# You can't reorder axes with reshape.
print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n") 

# This is a mess
print(tf.reshape(rank_3_tensor, [5, 6]), "\n")

# This doesn't work at all
try:
  tf.reshape(rank_3_tensor, [7, -1])
except Exception as e:
  #print(f"{type(e).__name__}: {e}")
  print("Eror, tidak harmonis tensor yang di reshape")

"""
kemungkinan ada nya suatu shapes yang panjang axisnya tidak diketahui ataupun rank tensor tidak diketahui
=>cara mengatasi masalahnya yakni dengan menerapkan tf.RangedTensor, yg mana shapes seperti hanya muncul dalam:

-tf.fungsi
-API fungsional dengan Keras
""" 
 #DTypes
"""
untuk mengetahui tipe data tf.tensor dapat dilakukan dengan menggunakan properti Tensor.dtype
tf.int32=>integer
tf.float32=>floating point
"""
tf32=tf.constant([[1.0,2.0],[3.0,2.0],[3.0,3.0]])
print(tf32)
tf64=tf.cast(tf32, dtype=tf.float64)
print(tf64)
#convert to int
int32=tf.cast(tf64, dtype=tf.int32)
print(int32)


#Penyiaran
x=tf.constant([[1],[2],[3]])
x=tf.reshape(x, [3,1])
y=tf.range(1,5)
print(x,'\n',y)
print(tf.multiply(x, y))#x nya di renggangkan secara otomatis agar pas dengan tensor yang lebih besar
print(x*y)
#atau bisa menggunakan tf.broadcast_to()
print(tf.broadcast_to(x,y))#tidak mengifisiensikan memory->lebih rumit

#tensor ragged
"""
untuk data yang tidak rata.dengan jumlah elemen beberapa axis

"""
ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]
try:
  tensor = tf.constant(ragged_list)
  print(tensor)
except Exception as e:
  print(f"{type(e).__name__}: {e}")
#tidak bisa dicetak dengan tensor biasa maka dari itu:
try:
  ragged_tensor=tf.ragged.constant(ragged_list)
  print(ragged_tensor)
  print(ragged_tensor.shape)#tidak diketahui jumlah nilai pada axisnya
except Exception as e:
   print(f"{type(e).__name__}: {e}")
   
#tensor string
"""
# If you have three string tensors of different lengths, this is OK.
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
"""
# If you have three string tensors of different lengths, this is OK.
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
print(tf.strings.split(tensor_of_strings),sep=' ')

#convert tensor to number
text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))#tensor di pisah pisah di setiap indexnya, setiap index string di convert ke number




#Tensor jarang
"""
mengoperasikan data yang bersifat jarang dengan penyematan sangat luas
dengan menggunakan tf.sparse.sparsetensor
untuk mengimplementasikannya bisa menggunakan tensor.sparse.to_dense(input)
"""
# Sparse tensors store values by index in a memory-efficient manner
sparse_tensor=tf.sparse.SparseTensor(indices=[[0,0],[1,2],[2,3]], 
                                     values=[1,2,3],
                                     dense_shape=[3,4])
print(sparse_tensor)
tensor=tf.sparse.to_dense(sparse_tensor)
print(tensor)