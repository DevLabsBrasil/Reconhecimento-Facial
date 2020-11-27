import cv2

class FaceDetect(object):

	def __init__(self, image="img.png"):

		"""
		image:str: Path de alguma imagem, padrão `img.png`
		"""

		self.img = cv2.imread(image)
		self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		self.clf = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
		self.eclf = cv2.CascadeClassifier("haarcascade_eye.xml") # Carrega o arquivo necessário.

	@property
	def face(self):
		return self.clf.detectMultiScale(self.gray, 1.3, 5)
	
	@property
	def printer(self):
		return "Faces Detectadas: %s" % len(self.face)
	
	def execute(self):

		print(self.printer) # Printa as faces detectadas ^

		for (x, y, w, h) in self.clf.detectMultiScale(self.gray, 1.3, 5):
			self.img = cv2.rectangle(self.img, (x, y), (x+w, y+h), (255, 255, 0), 2)

			for (ex, ey, ew, eh) in self.eclf.detectMultiScale(self.img[y:y+h, x:x+w]):
				cv2.rectangle(self.img[y:y+h, x:x+w], (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

	def run(self):

		self.execute()

		cv2.imshow("image", self.img)
		cv2.imwrite("image_detected.png", self.img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


if __name__ == '__main__':
	FaceDetect().run()
