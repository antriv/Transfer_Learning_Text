from squad.demo_prepro import prepro
from basic.demo_cli import Demo


class MachineComprehend:
	def __init__(self):
		self.model = Demo()

	def answer_question(self, paragraph, question):
		pq_prepro = prepro(paragraph, question)
		answer = self.model.run(pq_prepro)
		if len(answer) == 0:
			return None
		return answer


if __name__ == "__main__":
	import sys
	mc = MachineComprehend()
	p = open(sys.argv[1], 'r')
	paragraph = []
	for line in p:
		paragraph.append(line.rstrip())
	paragraph = ". ".join(paragraph)
	question = " ".join(sys.argv[2:])
	print(mc.answer_question(paragraph, question))

