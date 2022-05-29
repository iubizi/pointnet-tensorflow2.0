####################
# 读取测试
####################

import numpy as np

doc = np.load('ModelNet40.npz')

print('doc.files =', doc.files)
print(); print(); print()

for i in doc.files:

    print('doc[\'{}\'].shape ='.format(i), doc[i].shape)
    print('max(doc[\'{}\']) ='.format(i), np.max(doc[i]))
    print('min(doc[\'{}\']) ='.format(i), np.min(doc[i]))
    print('='*30)
    print(doc[i])
    print(); print(); print()
