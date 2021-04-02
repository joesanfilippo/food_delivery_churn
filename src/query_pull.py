import time
import requests
import json

def poll_job(s, redash_url, job):
    """ Example code adopted from https://github.com/getredash/redash-toolbelt/blob/master/redash_toolbelt/examples/refresh_query.py
    Args:
        s (Request Session): A session from HTTP requests library
        redash_url (str): Url of the Redash website to use
        job: json job returned from the session response

    Returns:
        None
        Polls Redash API for a query result
    """
    while job['status'] not in (3,4):
        response = s.get(f"{redash_url}/api/jobs/{job['id']}")
        job = response.json()['job']
        time.sleep(1)

    if job['status'] == 3:
        return job['query_result_id']
    
    return None

def get_fresh_query_result(redash_url, query_id, api_key, params):
    """ Example code adopted from https://github.com/getredash/redash-toolbelt/blob/master/redash_toolbelt/examples/refresh_query.py
    Args:
        redash_url (str): Url of the Redash website to use
        query_id (int): Unique identifier of the query stored in Redash system
        api_key (str): The Redash user's unique API key
        params (dict): A dictionary of optional parameters to use in the format key = Param Name, value = Param value

    Returns:
        None
        Polls Redash API for a query result
    """
    s = requests.Session()
    s.headers.update({'Authorization': f"Key {api_key}"})

    payload = dict(max_age=0, parameters=params)

    response = s.post(f"{redash_url}/api/queries/{query_id}/results", data = json.dumps(payload))

    if response.status_code != 200:
        raise Exception('Refresh failed.')

    result_id = poll_job(s, redash_url, response.json()['job'])

    if result_id:
        response = s.get(f"{redash_url}/api/queries/{query_id}/results/{result_id}.json")
        if response.status_code != 200:
            raise Exception('Failed getting results.')
    else:
        raise Exception('Query execution failed.')

    return response.json()['query_result']['data']['rows']

if __name__ == '__main__':
    
    print("Well hello there.")