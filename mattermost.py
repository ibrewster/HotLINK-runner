from datetime import datetime, timedelta

import mattermostdriver

import config
from utils import preevents_cursor, interpret_rections

def connect():
    mattermost = mattermostdriver.Driver(
        {
            "url": config.MATTERMOST_URL,
            "token": config.MATTERMOST_TOKEN,
            "port": config.MATTERMOST_PORT,
        }
    )

    mattermost.login()
    channel_id = mattermost.channels.get_channel_by_name_and_team_name(
        config.MATTERMOST_TEAM, config.MATTERMOST_CHANNEL
    )["id"]
    return (mattermost, channel_id)


def mm_upload(mattermost, channel_id, message, image=None, img_name=None):
    post_payload = {
        "channel_id": channel_id,
    }

    # First, upload the thumbnail, if any
    if image and img_name:
        image.seek(0)
        upload_result = mattermost.files.upload_file(
            channel_id=channel_id, files={"files": (img_name, image)}
        )

        matt_id = upload_result["file_infos"][0]["id"]
        post_payload["file_ids"] = [matt_id]

    if message:
        post_payload["message"] = message

    msg_meta = mattermost.posts.create_post(post_payload)
    return msg_meta

def get_channel_reactions():
    # TODO: Figure out how to do the filtering
    START_DATE = "2026-02-01 00:00:00"
    since_timestamp = int(datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    params = {'since': since_timestamp}
    mattermost, channel_id = connect()
    result = mattermost.posts.get_posts_for_channel(channel_id, params=params)

    posts = result.get('posts', {})
    order = result.get('order', [])
    votes = {}
    preevents_records = {}
    for post_id, post in posts.items():
        if post['user_id'] != config.MATTERMOST_USER_ID:
            continue

        reactions = set()
        post_reactions = post.get('metadata', {}).get('reactions', [])
        if post_reactions:
            for reaction in post_reactions:
                reactions.add(reaction['emoji_name'])
            post_creation = datetime.fromtimestamp(post['create_at'] / 1000)
            from_search = post_creation - timedelta(days=1)
            to_search = post_creation + timedelta(days=1)
            preevents_id = find_preevents_record_id(post_id, from_search, to_search)
            if not preevents_id:
                print("No preevents record found for record. Skipping.")
                continue

            preevents_records[post_id] = preevents_id

            true_pos, source = interpret_rections(reactions)
            save_reactions(preevents_id, true_pos, source)

            votes[post_id] = {
                'TruePos': true_pos,
                'Source': source,
            }

    for post_id,vote in votes.items():
        print(post_id, vote)

    print("------------------------------")

    for post_id, pid in preevents_records.items():
        print(post_id, pid)


def find_preevents_record_id(post_id, dfrom, dto):
    SQL ="""
    SELECT datavalue_id
    FROM datavalues
    WHERE categoryvalue->>'mattermostid'=%s
        AND datastream_id in (SELECT datastream_id
            FROM datastreams
            WHERE variable_id=13
            AND device_id=1)
        AND timestamp>=%s AND timestamp<%s;
    """
    with preevents_cursor() as cursor:
        cursor.execute(SQL, (post_id, dfrom, dto))
        result = cursor.fetchone()
    if result:
        return result[0]

def save_reactions(record_id, true_pos, source):
    print(f"Updating record {record_id} with values {true_pos}, {source}")
    with preevents_cursor(readonly = False, autocommit = True) as cursor:
        cursor.execute(
            """
            UPDATE datavalues
            SET categoryvalue = categoryvalue || jsonb_build_object(
                'true_positive', %s::boolean,
                'hotspot_source', %s::text
            )
            WHERE datavalue_id = %s
            """,
            (true_pos, source, record_id)
        )


if __name__ == "__main__":
    get_channel_reactions()
