import cv2

global frame
global point1, point2


# 截取需要查找的行人图片
def on_mouse(event, x, y, flags, param):
	global frame, point1, point2
	img2 = frame.copy()
	if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
		point1 = (x, y)
		cv2.circle(img2, point1, 10, (0, 255, 0), 5)
		cv2.imshow('image', img2)
	elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
		cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 5)
		cv2.imshow('image', img2)
	elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
		point2 = (x, y)
		cv2.rectangle(img2, point1, point2, (0, 0, 255), 5)
		cv2.imshow('image', img2)
		min_x = min(point1[0], point2[0])
		min_y = min(point1[1], point2[1])
		width = abs(point1[0] - point2[0])
		height = abs(point1[1] - point2[1])
		cut_img = frame[min_y:min_y + height, min_x:min_x + width]
		cv2.imwrite('query/0001_c1s1_0_%s.jpg'%min_x, cut_img)

if __name__ == '__main__':

	videoCapture = cv2.VideoCapture('center.mp4')
	# Read image
	success, frame = videoCapture.read()
	while success:
		cv2.namedWindow('image')
		cv2.setMouseCallback('image', on_mouse)
		cv2.imshow('image', frame)
		cv2.waitKey(0)
		success, frame = videoCapture.read()
