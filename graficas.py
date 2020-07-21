# -*- coding: utf-8 -*-
"""
Visualización de las gráficas de los losses

@author: Noelia Ubierna Fernández
"""

import csv
import matplotlib.pyplot as plt

csvfile = open('graficas.csv', 'r')

reader = csv.reader(filter(lambda row: row[0]!='#', csvfile))
                           
loss = []
rpn_class_loss = []
rpn_bbox_loss = []
mrcnn_class_loss = []
mrcnn_bbox_loss = []
mrcnn_mask_loss = []

steps = []

# Se accede a cada registro del fichero
for i, row in enumerate(reader):
    loss.append(float(row[0]))
    rpn_class_loss.append(float(row[1]))
    rpn_bbox_loss.append(float(row[2]))
    mrcnn_class_loss.append(float(row[3]))
    mrcnn_bbox_loss.append(float(row[4]))
    mrcnn_mask_loss.append(float(row[5]))
    steps.append(i)
    

# Se cierra el fichero    
csvfile.close()

#loss
fig, ax1 = plt.subplots()
ax1.plot(steps, loss, "m-", label="loss")
ax1.set_xlabel("steps")
ax1.set_ylabel("loss")
plt.title("loss")
plt.savefig("grafica_loss.png")

#rpn_class_loss
fig, ax2 = plt.subplots()
ax2.plot(steps, rpn_class_loss, "r-", label="rpn_class_loss")
ax1.set_xlabel("steps")
ax1.set_ylabel("rpn_class_loss")
plt.title("rpn_class_loss")
plt.savefig("grafica_rpn_class_loss.png")

#rpn_bbox_loss
fig, ax3 = plt.subplots()
ax3.plot(steps, rpn_bbox_loss, "g-", label="rpn_bbox_loss")
ax1.set_xlabel("steps")
ax1.set_ylabel("rpn_bbox_loss")
plt.title("rpn_bbox_loss")
plt.savefig("grafica_rpn_bbox_loss.png")

#mrcnn_class_loss
fig, ax4 = plt.subplots()
ax4.plot(steps, mrcnn_class_loss, "b-", label="mrcnn_class_loss")
ax1.set_xlabel("steps")
ax1.set_ylabel("mrcnn_class_loss")
plt.title("mrcnn_class_loss")
plt.savefig("grafica_mrcnn_class_loss.png")

#mrcnn_bbox_loss
fig, ax5 = plt.subplots()
ax5.plot(steps, mrcnn_bbox_loss, "y-", label="mrcnn_bbox_loss")
ax1.set_xlabel("steps")
ax1.set_ylabel("mrcnn_bbox_loss")
plt.title("mrcnn_bbox_loss")
plt.savefig("grafica_mrcnn_bbox_loss.png")

#mrcnn_mask_loss
fig, ax6 = plt.subplots()
ax6.plot(steps, mrcnn_mask_loss, "k-", label="mrcnn_mask_loss")
ax1.set_xlabel("steps")
ax1.set_ylabel("mrcnn_mask_loss")
plt.title("mrcnn_mask_loss")
plt.savefig("grafica_mrcnn_mask_loss.png")

plt.plot()

