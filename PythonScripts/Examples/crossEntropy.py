# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:30:59 2023

@author: Genio
"""

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(1e-4, 1, 1000)  # Sample data.
y = np.log(x)

styles = plt.style.available

plt.close('all')

fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
plt.style.use(styles[13])
ax.plot(x, y, label='ln(x)') 
ax.plot(x,-y, label='-ln(x)') 
ax.set_xlabel('a')  
ax.set_ylabel('CE')
ax.set_title("$CrossEntropy: CE = -[y \ln(a)+ (1-y) \ln(1-a)]$")
ax.axis([-0.1, 1.1, -10, 10])
ax.grid(True)
ax.legend()
plt.show()

fig.savefig('CrossEntropy.png', dpi=300)
