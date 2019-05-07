"""Model analysis tools."""


def print_maxent_features(vect, clf, n=5):
	"""
	Most relevant features for each class (logistic regression).

	vect -- vectorizer (count or tf-idf)
	clf -- LogisticRegression classifier
	n -- number of features to show
	"""
	C = clf.coef_
	A = clf.coef_.argsort()
	features = vect.get_feature_names()
	for i, label in enumerate(clf.classes_):
		print('{}:'.format(label))
		print('\t{} ({})'.format(
			' '.join([features[j] for j in A[i, :n]]),
			C[i, A[i, :n]]))
		print('\t{} ({})'.format(
			' '.join([features[j] for j in A[i, -n:]]),
			C[i, A[i, -n:]]))

def maxent_features_to_html(vect, clf, n=5):
	template = '<html>\n<head></head>\n<body>{body}</body></html>'
	tableTemplate = '<table><thead>\
		<tr><th>Token</th><th>Peso</th></tr></thead><tbody>\
		{tableBody}</tbody></table>'
	rowTemplate = '<tr><td>{token}</td><td>{weight:.4f}</td></tr>'
	C = clf.coef_
	A = clf.coef_.argsort()
	features = vect.get_feature_names()
	body = ''
	for i, label in enumerate(clf.classes_):
		tables = ''
		for r in [slice(0, n), slice(-1, -n, -1)]:
			tb = ''
			for j in A[i, r]:
				t = features[j]
				w = C[i, j]
				row = rowTemplate.format(token=t, weight=w)
				tb += row
			table = tableTemplate.format(tableBody=tb)
			tables += table
		body += f'<h4>{label}</h4><div class="ftables">{tables}</div>'
	html = template.format(body=body)

	return html

	

def print_feature_weights_for_item(vect, clf, x):
	"""
	Print active features and their weight for a specific item.

	vect -- text vectorizer (count or tf-idf)
	clf -- LogisticRegression classifier
	"""
	features = vect.get_feature_names()
	x2 = vect.transform([x])
	col = x2.tocoo().col
	for i in col:
		print(features[i], clf.coef_[:,i])
