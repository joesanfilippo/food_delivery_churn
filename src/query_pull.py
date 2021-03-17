import time
import requests

def poll_job(s, redash_url, job):
    while job['status'] not in (3,4):
        response = s.get(f"{redash_url}/api/jobs/{job['id']}")
        job = response.json()['job']
        time.sleep(1)

    if job['status'] == 3:
        return job['query_result_id']
    
    return None

def get_fresh_query_result(redash_url, query_id, api_key, params):
    s = requests.Session()
    s.headers.update({'Authorization': f"Key {api_key}"})

    response = s.post(f"{redash_url}/api/queries/{query_id}/refresh", params=params)

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