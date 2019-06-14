from __future__ import print_function # Python 2/3 compatibility
import boto3
import json
import decimal
from boto3.dynamodb.conditions import Key, Attr

# Helper class to convert a DynamoDB item to JSON.
class DecimalEncoder(json.JSONEncoder):
	def default(self, o):
		if isinstance(o, decimal.Decimal):
			if abs(o) % 1 > 0:
				return float(o)
			else:
				return int(o)
		return super(DecimalEncoder, self).default(o)

class DecimalEncoder_str(json.JSONEncoder):
	def default(self, o):
		if isinstance(o, decimal.Decimal):
			return str(o)
		return super(DecimalEncoder, self).default(o)


dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

table = dynamodb.Table('SamsaraSD')

def ddb_add(classification, url, annotation):

	try:
		response = table.put_item(
			Item={
				'URL': url,
				'Classification': decimal.Decimal(classification),
				'Annotation': decimal.Decimal(annotation)
			}
		)
	except ClientError as e:
		print(e.response['Error']['Message'])
		return 0
	else:
		print("PutItem succeeded:")
		print(json.dumps(response, indent=4, cls=DecimalEncoder))
		return 1

def ddb_read(url):
	try:
		response = table.get_item(
			Key={
				'year': year,
				'title': title
			}
		)
	except ClientError as e:
		print(e.response['Error']['Message'])
	else:
		item = response['Item']
		print("GetItem succeeded:")
		print(json.dumps(item, indent=4, cls=DecimalEncoder))

def ddb_query_URL(url):

	response = table.query(
		ProjectionExpression="#cl",
		ExpressionAttributeNames={ "#cl": "Classification" }, # Expression Attribute Names for Projection Expression only.
		KeyConditionExpression=Key('URL').eq(url)
	)

	for i in response[u'Items']:
		print(json.dumps(i, cls=DecimalEncoder_str))

def ddb_update(classification, url, annotation):

	response = table.update_item(
		Key={'URL': url
		},
		UpdateExpression="set Classification = :c, set Annotation = :a",
		ExpressionAttributeValues={
			':c': decimal.Decimal(classification),
			':a': decimal.Decimal(annotation)
		},
		ReturnValues="UPDATED_NEW"
	)
	print("UpdateItem succeeded:")
	print(json.dumps(response, indent=4, cls=DecimalEncoder))
