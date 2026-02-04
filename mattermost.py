import urllib
import mattermostdriver

import config

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

    mattermost.posts.create_post(post_payload)